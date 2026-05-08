# EDMolGPT: From Holo Pockets to Electron Density

**Paper Title:** From Holo Pockets to Electron Density: GPT-style Drug Design with Density  
**Published at:** ICML 2026

## Abstract

Recent advances in generative modeling have enabled significant progress in structure-based drug design (SBDD). Existing methods typically condition molecule generation on empty binding pockets from holo complexes, overlooking informative components such as the filler (ligands and solvent). Here, we leverage low-resolution electron density (ED) derived from the filler as a physically grounded condition for *de novo* drug design. We consider two types of ED—calculated and cryo-EM/X-ray—obtainable from computational or experimental sources, supporting unified pre-training and experimental integration. Compared with rigid pocket representations, experimental ED naturally captures conformational flexibility and provides a more faithful description of the binding environment. Based on this, we introduce EDMolGPT, a decoder-only autoregressive framework that generates molecules from low-resolution ED point clouds. By grounding generation in physically meaningful density signals, EDMolGPT mitigates structural bias and produces molecules with 3D conformations. Evaluations on 101 biological targets verify the effectiveness.

## Overview

EDMolGPT is a GPT-style autoregressive model for drug design that uses electron density (ED) as a physical condition. The model takes low-resolution ED point clouds as input and generates 3D molecules with conformational information, addressing limitations of traditional rigid pocket representations.

Key features:
- Generates molecules from electron density point clouds
- Supports both calculated and experimental (cryo-EM/X-ray) ED data
- Captures conformational flexibility of binding pockets
- Mitigates structural bias in drug design
- Generates molecules with 3D conformations

## Installation

### Requirements

The project requires Python 3.9+ and several dependencies. We recommend using Conda for installation.

### Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f mole.yaml
conda activate mole

```

### Manual Installation

```bash
# Install PyTorch (with GPU support if available)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install RDKit
pip install rdkit-pypi

# Install other dependencies
pip install transformers accelerate numpy pandas scipy matplotlib tqdm pyyaml
```

## Usage

### Inference

To run inference on the provided validation set:

```bash
python main.py --gpu 0
```

This will:
1. Load the pre-trained model (from `./9_31454_4.pt`)
2. Process electron density files from `./valset_large/`
3. Generate molecules for each target


### Parameters

- `--gpu`: GPU device ID to use (default: -1 for CPU)

### Output

Generated molecules will be saved as `.mol` files in the `./outputs` directory, organized by target.

## Project Structure

```
EDMolGPT/
├── main.py              # Main inference script
├── mole.yaml            # Conda environment configuration
├── README.md            # This file
├── valset_large/        # Validation set with target structures and ED data
├── dataloader/          # Data loading utilities
│   ├── dataloader_pointcloud.py    # Point cloud data loader
│   └── data_utils.py               # Data reading and processing functions
├── model/
│   └── gpt2.py          # GPT2-based model architecture
├── util/                # Utility functions
│   ├── ligand_code_util.py    # Ligand encoding/decoding
│   ├── find_root.py          # Root finding algorithm
│   └── fragmol_frag_zyh.py   # Fragment-based molecule generation
└── 9_31454_4.pt         # Pre-trained model checkpoint
```

## Model Architecture

EDMolGPT extends the GPT2 architecture to incorporate electron density information. Key components:

1. **ED Encoding**: Low-resolution ED point clouds are encoded using symbol and position embeddings
2. **Resolution Embedding**: Embeds ED resolution as additional information
3. **Coordinate Prediction**: Predicts 3D coordinates using positional mapping layers
4. **Bond/Angle Prediction**: Predicts bond lengths, angles, and dihedral angles
5. **Autoregressive Generation**: Generates molecules token by token with structural constraints

## Data Format

The validation set includes:
- `.sdf` files: Known ligand structures
- `_resX.pdb` files: Electron density point clouds at different resolutions (1.5Å, 2.7Å, 3.5Å, 5.0Å, 8.0Å)

The model is trained on a larger dataset of protein-ligand complexes with corresponding ED maps.
