import os
import numpy as np
import torch.utils.data as data
import torch
from rdkit import Chem

import random

from tqdm import tqdm
# from rdkit.Chem import rdchem

from util.ligand_code_util import LigandCodeUtil 

from .data_utils import ed_pdb_read
# import random
import matplotlib.pyplot as plt

import csv

from openbabel import pybel
import copy
from rdkit.Chem import AllChem, rdGeometry


def single_conf_gen(tgt_mol, num_confs=1, seed=42):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed, clearConfs=True)
        sz = len(allconformers)
        for i in range(sz):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            except:
                continue
        mol = Chem.RemoveHs(mol)
        return mol

def get_mol_center(mol):
    """
    计算分子的几何中心（简单平均，不考虑原子质量）
    """
    conf = mol.GetConformer()
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    mol_range = (positions.max(axis=0) + positions.min(axis=0)) / 2
    return mol_range


class PointCloud(data.Dataset):
    def __init__(self, root='', train=True):
        files = os.listdir(root)

        self.sdf = []
        self.point_cloud = {}
        self.names = []
        
        self.symbol_point_cloud = {'N':290, 'O':291, 'F':292, 'C':293}
        self.fragUtil = LigandCodeUtil(scale=28, resolution=0.1, max_token_length=180)
        error = []
        self.train = train

        for file in tqdm(files):
            if 'sdf' in file:
                name = file[:-4]
                # if name in error:
                #     # print('skip')
                #     continue
                self.names.append(name)
                self.sdf.append(os.path.join(root, file))
                self.point_cloud[os.path.join(root, file)] = [os.path.join(root, name + '_res1.5.pdb'), 
                                          os.path.join(root, name + '_res2.7.pdb'),
                                          os.path.join(root, name + '_res3.5.pdb'),
                                          os.path.join(root, name + '_res5.0.pdb'),
                                          os.path.join(root, name + '_res8.0.pdb')]
                # print(os.path.join(root, file))
        self.root = root
    def __getitem__(self, index):
        # while True:
        #     try:
        sdf_file = self.sdf[index]
        # print(sdf_file)
        
        mol =  Chem.SDMolSupplier(sdf_file)[0]
        # start_idx = random.choice([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 6])
        # _ = Chem.MolToSmiles(mol, rootedAtAtom=int(start_idx))
        # mol = Chem.RenumberAtoms(
        #     mol,
        #     mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"],
        # )
        # mol = Chem.MolFromSmiles(sdf_file.split('/')[-1][:-4])
        # mol = single_conf_gen(mol, num_confs=1)

        if 'dude' in self.root or 'molcraft' in self.root:
            center = get_mol_center(mol)
        else:
            center = np.array([0, 0, 0])
        # print(center)
        code_outputs = self.fragUtil.encode_mol(mol, center, debug=False)
        mol_code, mol_pos = code_outputs['ligand_tokens'], code_outputs['ligand_coords']
        ligand_r1_indices, ligand_r2_indices, ligand_r3_indices = code_outputs['ligand_r1_indices'], code_outputs['ligand_r2_indices'], code_outputs['ligand_r3_indices']
        bond_angles, bond_lengths = code_outputs['bond_angles'], code_outputs['bond_lengths']
        dihedral_angles = code_outputs['dihedral_angles']

        point_cloud_files = self.point_cloud[sdf_file]
        if not self.train:
            resolution=2
            # resolution=0
            print(point_cloud_files[resolution])
        else:
            resolution = random.choice([i for i in range(5)])
        point_cloud_file = point_cloud_files[resolution]


        pc_code, pc_pos = ed_pdb_read(point_cloud_file)
        pc_code = [self.symbol_point_cloud[pc_code[i]] for i in range(len(pc_code))]
        # pc_code = [self.symbol_point_cloud['C'] for i in range(len(pc_code))]
        pc_pos = self.convert_pos_pointcloud(pc_pos, center)

        pc_pos = torch.LongTensor(pc_pos) + 1
        mol_pos = torch.LongTensor(mol_pos) + 1

        pc_code = torch.LongTensor(pc_code)
        mol_code = torch.LongTensor(mol_code)

        ligand_r1_indices = torch.LongTensor(ligand_r1_indices)
        ligand_r2_indices = torch.LongTensor(ligand_r2_indices)
        ligand_r3_indices = torch.LongTensor(ligand_r3_indices)

        bond_angles = torch.LongTensor(bond_angles)
        bond_lengths = torch.LongTensor(bond_lengths)

        dihedral_angles = torch.LongTensor(dihedral_angles)

        # s_idx = torch.randperm(pc_code.size(0))
        # pc_pos = pc_pos[s_idx]
        # pc_code = pc_code[s_idx]


            #     break
            # except KeyboardInterrupt:
            #     assert 0
            # except:
            #     with open('error3', 'a') as error_file:
            #         error_file.write(self.sdf[index].strip() + '\n')
            #     index = random.randint(0, len(self) - 1)
            #     print('select another sample!')
                
        begin_pc_pos, end_pc_pos = torch.LongTensor([[299, 299, 299]]), torch.LongTensor([[299, 299, 299]])
        begin_pc_type, end_pc_type = torch.LongTensor([294]), torch.LongTensor([295])
        pc_pos = torch.cat([begin_pc_pos, pc_pos, end_pc_pos], dim=0)
        pc_code = torch.cat([begin_pc_type, pc_code, end_pc_type], dim=0)
        
        return {'pointcloud':[pc_pos, pc_code, torch.LongTensor([resolution])], 
                'mol':[mol_pos, mol_code, ligand_r1_indices, ligand_r2_indices, ligand_r3_indices, bond_angles, bond_lengths, dihedral_angles],
                'name':sdf_file,
                'smiles':self.names[index],
                'cliques':code_outputs['cliques'],
                'atoms_ids':code_outputs['atoms_ids'],
                'frags':code_outputs['frags'],
                'center':center,
                'point_cloud_file':point_cloud_file}
    
    def __len__(self):
        return len(self.sdf)

    def convert_pos_pointcloud(self, pc_pos, center=np.array([0, 0, 0])):
        grid_size = (self.fragUtil.grid_size - 1) / 2
        # center = np.array([0, 0, 0])

        pc_pos = (np.array(pc_pos) - center) / self.fragUtil.resolution + np.array([grid_size, grid_size, grid_size])
        pc_pos = np.rint(pc_pos).astype("int")

        for i in range(len(pc_pos)):
            pos = pc_pos[i]
            if np.min(np.array(pos)) < 0 or np.max(np.array(pos)) >= self.fragUtil.grid_size:
                pc_pos[i] = [-1, -1, -1]

        return pc_pos

    def invert(self, pos, center=np.array([0, 0, 0])):
        grid_size = (self.fragUtil.grid_size - 1) / 2
        # center = np.array([0, 0, 0])
        pos = (pos - np.array([grid_size, grid_size, grid_size])) * self.fragUtil.resolution + center
        return pos
