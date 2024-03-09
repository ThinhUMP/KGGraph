import torch
import numpy as np
from rdkit import Chem
from pathlib import Path
import sys
import pandas as pd
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))

from KGGraph.Chemistry.chemutils import get_atom_types
from KGGraph.MotifGraph.motif_decompose import MotifDecomposition

# Load data and get atom types
#TODO: Modify atom_types when adding MoleculeDataset class
data = pd.read_csv('./data/testcase_featurize.csv')
smiles = data['SMILES'].tolist()
atom_types = get_atom_types(smiles)

# Get supernode features
x_supernode = torch.tensor([0]*len(atom_types) + [0, 1], dtype=torch.long)



if __name__ == '__main__':
    print(x_supernode)