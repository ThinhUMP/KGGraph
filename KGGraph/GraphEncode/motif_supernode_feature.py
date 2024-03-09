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

from KGGraph.Chemistry.chemutils import get_atom_types, get_mol
from KGGraph.MotifGraph.MotitDcp.motif_decompose import MotifDecomposition
from KGGraph.GraphEncode.atom_feature import AtomFeature

# Load data and get atom types
#TODO: Modify atom_types when adding MoleculeDataset class
data = pd.read_csv('./data/testcase_featurize.csv')
smiles = data['SMILES'].tolist()
atom_types = get_atom_types(smiles)

def motif_supernode_feature(mol: Chem.Mol, atom_types):
    """
    Compute motif and supernode features for a given molecule.
    
    Parameters:
        mol: The input molecule.
        atom_types: The list of atom types.
        
    Returns:
        A tuple of tensors representing motif and supernode features.
    """
    number_atom_node_attr = 42
    motif = MotifDecomposition()
    cliques = motif.defragment(mol)
    num_motif = len(cliques)

    # Pre-define tensor templates
    supernode_template = [0] * len(atom_types) + [0, 1] + [0] * (number_atom_node_attr - len(atom_types) - 2)
    motif_node_template = [0] * len(atom_types) + [1, 0] + [0] * (number_atom_node_attr - len(atom_types) - 2)

    # Create tensors based on the number of motifs
    x_supernode = torch.tensor([supernode_template], dtype=torch.long)
    if num_motif > 0:
        x_motif = torch.tensor([motif_node_template] * num_motif, dtype=torch.long)
    else:
        x_motif = torch.empty(0, number_atom_node_attr, dtype=torch.long)  # Handle cases with no motifs

    return x_motif, x_supernode

def x_feature(mol: Chem.Mol, atom_types):
    """
    Compute the feature vector for a given molecule.
    
    Parameters:
        mol: The input molecule.
        atom_types: The list of atom types.
        
    Returns:
        A tensor representing the feature vector.
    """
    atom_feature = AtomFeature(mol=mol)
    x_node_atom = atom_feature.feature()
    x_motif, x_supernode = motif_supernode_feature(mol, atom_types)

    # Concatenate features
    x = torch.cat((x_node_atom, x_motif.to(x_node_atom.device), x_supernode.to(x_node_atom.device)), dim=0)
    return x

if __name__ == '__main__':
    mol = get_mol(smiles[0])
    x = x_feature(mol, atom_types)
    print(x.size())  # Print the size of the feature vector
