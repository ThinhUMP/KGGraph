# KGG
Knowledge-Guided Graph Self-Supervised Learning to Enhance Molecular Property Predictions [(ChemRxiv)](https://chemrxiv.org/engage/chemrxiv/article-details/68088e84e561f77ed461ef2d)

## Overview

**Knowledge‑Guided Graph (KGG)** is a lightweight, self‑supervised pre‑training framework that injects orbital‑level chemical knowledge into Graph Neural Networks (GNNs) for molecular property prediction.

### Why KGG?
- **Orbital‑aware descriptors** – Hybridization‑ and bond‑type vectors embed explicit orbital information for richer chemical context. 
- **Data‑efficient pre‑training** – Self‑supervised on ~250 k ZINC15 molecules (≈10 × less than typical), easing label scarcity and lowering contamination ratio.  
- **Plug‑and‑play compatibility** – Works out of the box with popular GNN backbones (GIN, GCN, GAT, GraphSAGE).  
- **State‑of‑the‑art accuracy** – Consistently surpasses existing methods across diverse molecular property benchmarks, especially with noisy data.  
- **Chemically interpretable embeddings** – t‑SNE and fingerprint analyses show clear, domain‑aligned clustering.  


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Publication](#publication)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Installation

To install and set up the KGG framework, follow these steps. 

1. **Step by step installation**

**Creating a Virtual Environment (Optional but Recommended):**
  It's recommended to use a virtual environment to avoid conflicts with other projects or system-wide packages. Use the following commands to create and activate a virtual environment:

```bash
# Create and activate a new Conda environment with Python 3.11
conda create -n kgg python=3.11
conda activate kgg

# Install torch and rdkit packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge rdkit

# Install PyTorch Geometric and dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install torch_geometric

# Install utility libraries
pip install joblib scikit-learn pytest black
```

2. **Install from environment.yml:**
```bash
conda env create -f environment.yml
```

## Usage

## Publication

[SynTemp: Efficient Extraction of Graph-Based Reaction Rules from Large-Scale Reaction Databases](https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01795)


### Citation
```
@article{to2025kgg,
  title={KGG: Knowledge-Guided Graph Self-Supervised Learning to Enhance Molecular Property Predictions},
  author={To, Van-Thinh and Van-Nguyen, Phuoc-Chung and Truong, Gia-Bao and Phan, Tuyet-Minh and Phan, Tieu-Long and Fagerberg, Rolf and Stadler, Peter and Truong, Tuyen},
  year={2025}
}
```


## Contributing
- [Van-Thinh TO](https://github.com/ThinhUMP)

## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

This work has received support from the Korea International Cooperation Agency (KOICA) under the project entitled “Education and Research Capacity Building Project at University of Medicine and Pharmacy at Ho Chi Minh City”, conducted from 2024 to 2025 (Project No. 2021-00020-3).