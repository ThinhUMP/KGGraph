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

1. Step by step installation

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


3. **Install from environment.yml:**
conda env create -f environment.yml


## Usage

### Use in script
  ```python
  from syntemp.auto_template import AutoTemp

  data = [{'R-id': 0,
    'reactions': 'COC(=O)C(CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O.O>>COC(=O)C(CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O.O=C(O)OCc1ccccc1'},
  {'R-id': 1,
    'reactions': 'Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O.O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1'},
  {'R-id': 4,
    'reactions': 'CCOc1ccc(Oc2ncnc3c2cnn3C2CCNCC2)c(F)c1.O=C(Cl)OC1CCCC1>>CCOc1ccc(Oc2ncnc3c2cnn3C2CCN(C(=O)OC3CCCC3)CC2)c(F)c1.Cl'},
  {'R-id': 5,
    'reactions': 'Cn1cnc(-c2cc(C#N)ccn2)c1Br.OB(O)c1ccc(-n2cccn2)cc1>>Cn1cnc(-c2cc(C#N)ccn2)c1-c1ccc(-n2cccn2)cc1.OB(O)Br'},
  {'R-id': 6,
    'reactions': 'CC1(C)OB(c2ccc(OCc3ccc4ccccc4n3)cc2)OC1(C)C.N#Cc1ccc(OC2CCCCO2)c(Br)c1>>CC1(C)OB(Br)OC1(C)C.N#Cc1ccc(OC2CCCCO2)c(-c2ccc(OCc3ccc4ccccc4n3)cc2)c1'}]

  auto = AutoTemp(
      rebalancing=True,
      mapper_types=["rxn_mapper", "graphormer", "local_mapper"],
      id="R-id",
      rsmi="reactions",
      n_jobs=1,
      verbose=2,
      batch_size=1,
      job_timeout=None,
      safe_mode=False,
      save_dir=None,
      fix_hydrogen=True,
  )

  (reaction_dicts, templates, hier_templates,
  its_correct, uncertain_hydrogen,) = auto.temp_extract(data, lib_path=None)

  core_temp = templates[0]



  # In order to perform forward prediction, you can use the following code:
  from synkit.IO import gml_to_its
  from synkit.Synthesis.Reactor.syn_reactor import SynReactor
  reactor = SynReactor(substrate='COC(=O)C(CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O.O', template=gml_to_its(core_temp[0]['gml']))

  print(reactor.smarts)
  ```
  

### Use in command line
  ```bash
  echo -e "R-id,reaction\n0,COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O>>COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O" > test.csv
  python -m syntemp --data_path test.csv --rebalancing --id 'R-id' --rsmi 'reaction' --rerun_aam --fix_hydrogen --log_file ./log.txt --save_dir ./
  ```

### Reproduce templates extraction
  Run these commands from the root of the cloned repository.
  ```bash
  python -m syntemp --data_path Data/USPTO_50K_original.csv --log_file Data/Test/log.txt --save_dir Data/Test/ --rebalancing --fix_hydrogen --rerun_aam --n_jobs 3 --batch_size 1000 --rsmi reactions --id ID
  ```
    
## Publication

[SynTemp: Efficient Extraction of Graph-Based Reaction Rules from Large-Scale Reaction Databases](https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01795)


### Citation
```
@article{phan2025syntemp,
  title={SynTemp: Efficient Extraction of Graph-Based Reaction Rules from Large-Scale Reaction Databases},
  author={Phan, Tieu-Long and Weinbauer, Klaus and Laffitte, Marcos E Gonz{\'a}lez and Pan, Yingjie and Merkle, Daniel and Andersen, Jakob L and Fagerberg, Rolf and Flamm, Christoph and Stadler, Peter F},
  journal={Journal of Chemical Information and Modeling},
  volume={65},
  number={6},
  pages={2882--2896},
  year={2025},
  publisher={ACS Publications}
}
```


## Contributing
- [Tieu-Long Phan](https://tieulongphan.github.io/)

## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

This project has received funding from the European Unions Horizon Europe Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No 101072930 ([TACsy](https://tacsy.eu/) -- Training Alliance for Computational)