#!/usr/bin/env python
# coding=utf-8
"""
EDMolGPT: From Holo Pockets to Electron Density - Inference Script

This script performs inference using the EDMolGPT model to generate molecules
from electron density point clouds.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.notebook import tqdm
from rdkit import Chem

from model.gpt2 import GPT2LMHeadModel, GPT2Config


def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def infer_case(
    root_dir="./valset_large",
    num_gpu=-1,
    model_path="./9_31454_4.pt",
    save_dir="./dude_exped",
    num_generations=100,
    max_retries=500
):
    """
    Run inference to generate molecules from electron density point clouds.

    Args:
        root_dir: Directory containing validation data
        num_gpu: GPU device ID (-1 for CPU)
        model_path: Path to pre-trained model checkpoint
        save_dir: Directory to save generated molecules
        num_generations: Number of molecules to generate per target
        max_retries: Maximum number of retry attempts per molecule
    """
    from dataloader.dataloader_pointcloud import PointCloud
    from dataloader.data_utils import read_point_cloud_gen, ed_pdb_read

    # Set device
    device = f"cuda:{num_gpu}" if num_gpu >= 0 else "cpu"

    # Initialize model
    config = GPT2Config(n_layer=24, n_embd=1024, n_ctx=1024, n_head=16)
    model = GPT2LMHeadModel(config).to(device)
    model.eval()

    # Load checkpoint
    state_dicts = torch.load(model_path, map_location="cpu")
    new_dicts = {}
    for k, v in state_dicts.items():
        new_k = k[7:]  # Remove 'module.' prefix if needed
        new_dicts[new_k] = v
    model.load_state_dict(new_dicts)

    # Load dataset
    val_set = PointCloud(root=root_dir, train=False)

    # Get all PDB files in the directory
    files = os.listdir(root_dir)
    res = []

    # Process each PDB file
    for file in files:
        p = os.path.join(root_dir, file)
        if "sdf" in p:
            continue  # Skip SDF files

        pc_code, pc_pos = ed_pdb_read(p)
        pc_code = [val_set.symbol_point_cloud[pc_code[i]] for i in range(len(pc_code))]
        pc_code = np.array(pc_code)
        pc_pos = np.array(pc_pos)

        # Sort points for consistency
        index = np.lexsort((pc_pos[:, 1], pc_pos[:, 0]))
        pc_code = pc_code[index]
        pc_pos = pc_pos[index]

        res.append((file, pc_code, pc_pos))

    # Create output directory if needed
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Generate molecules for each target
    for i in range(len(res)):
        file, pc_code, pc_pos = res[i]
        print(f"Processing {file}")
        num_pc_points = pc_code.shape[0] + 2

        center = np.mean(pc_pos, axis=0)
        pc_code = torch.LongTensor(pc_code)
        pc_pos = val_set.convert_pos_pointcloud(pc_pos, center)
        pc_pos = torch.LongTensor(pc_pos) + 1

        # Add special tokens for beginning/end
        begin_pc_pos, end_pc_pos = torch.LongTensor([[299, 299, 299]]), torch.LongTensor([[299, 299, 299]])
        begin_pc_type, end_pc_type = torch.LongTensor([294]), torch.LongTensor([295])
        pc_pos = torch.cat([begin_pc_pos, pc_pos, end_pc_pos], dim=0)
        pc_code = torch.cat([begin_pc_type, pc_code, end_pc_type], dim=0)

        resolution = torch.LongTensor([2])

        # Initialize with dummy values
        input_bond_angles = torch.ones((1, pc_pos.shape[0])).to(device).long() * (config.num_bond_ang - 1)
        input_bond_lengths = torch.ones((1, pc_pos.shape[0])).to(device).long() * (config.num_bond_leng - 1)
        input_dihedral_angles = torch.ones((1, pc_pos.shape[0])).to(device).long() * (config.num_dih_ang - 1)
        input_ligand_r1_indices = torch.zeros((1, pc_pos.shape[0])).to(device).long()

        # Generate molecules with retries
        attempt = 0
        num_success = 0

        while num_success != num_generations:
            attempt += 1
            setup_seed(attempt)

            if attempt > max_retries:
                break  # Stop after max retries

            try:
                # Generate using model
                gen_tokens, gen_pos, gen_bond_angle, gen_bond_len, gen_dih_angle, n1, n2, n3 = model.predict(
                    pc_code[None].to(device),
                    pc_pos[None].to(device),
                    resolution[None].to(device),
                    input_ligand_r1_indices.to(device),
                    input_bond_angles.to(device),
                    input_bond_lengths.to(device),
                    input_dihedral_angles.to(device),
                    fragmentutil=val_set.fragUtil,
                    shift=num_pc_points,
                )

                # Decode and save generated molecule
                gen_tokens = gen_tokens[0, num_pc_points:]
                gen_pos = (gen_pos[0, num_pc_points:] - 1).cpu().detach().numpy()
                gen_pos = val_set.invert(gen_pos, center)

                outputs = val_set.fragUtil.decode_3d(
                    gen_tokens.cpu().detach().numpy().reshape([1, -1]),
                    gen_pos[None, :, :],
                    debug=False
                )

                # Create subdirectory for this target
                suffix = file.split("_res")[0] if "_res" in file else file
                target_dir = os.path.join(save_dir, suffix)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)

                # Save if valid molecule
                if outputs["mols"][0] is not None:
                    mol_path = os.path.join(target_dir, f"gen_{num_success:03d}.mol")
                    Chem.MolToMolFile(Chem.RemoveHs(outputs["mols"][0]), mol_path)
                    num_success += 1

            except Exception as e:
                print(f"Generation failed: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDMolGPT Inference")
    parser.add_argument("--gpu", help="GPU device ID (-1 for CPU)", type=int, default=-1)
    parser.add_argument("--model", help="Path to model checkpoint", type=str, default="")
    parser.add_argument("--input", help="Input directory with ED data", type=str, default="")
    parser.add_argument("--output", help="Output directory for generated molecules", type=str, default="./outputs")
    parser.add_argument("--num", help="Number of molecules to generate per target", type=int, default=100)

    args = parser.parse_args()

    infer_case(
        root_dir=args.input,
        num_gpu=args.gpu,
        model_path=args.model,
        save_dir=args.output,
        num_generations=args.num
    )
