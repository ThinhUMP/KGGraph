import sys
import json
import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem
from typing import Tuple

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
    
# Import necessary modules and functions
from KGGraph.Chemistry.chemutils import get_mol
from KGGraph.Chemistry.features import (
    get_bond_type, is_conjugated, is_rotatable, get_stereo, get_bond_polarity, is_bond_in_ring
)

# Load bond dictionaries for process of feature extraction.
with open(root_dir / 'data/bond_dict.json', 'r') as f:
    bond_dict = json.load(f)

with open(root_dir / 'data/bond_stereo_dict.json', 'r') as f:
    bond_stereo_dict = json.load(f)

def edge_feature(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the edge features for the molecule.

    Returns:
        edge_attr_list: A tensor of edge attributes.
        edges_index_list: A tensor of edge indices.
    """
    num_bond_features = 32
    if len(mol.GetAtoms()) > 0:
    # Initialize lists to store edge attributes and indices
        edge_attr_list = []
        edges_index_list = []
        
        # Iterate over all bonds in the molecule
        for bond in mol.GetBonds():
            # Compute basic features for the bond
            basic_features = [
                is_conjugated(bond),
                is_rotatable(bond),
                get_bond_polarity(bond),
                is_bond_in_ring(bond)
            ]
            
            # Get bond type and stereo features from the dictionaries
            bond_type_feature = bond_dict.get(get_bond_type(bond), [0] * len(bond_dict))
            bond_stereo_feature = bond_stereo_dict.get(get_stereo(bond), [0] * len(bond_stereo_dict))
            
            # Combine all features into a single list
            combined_features = basic_features + bond_type_feature + bond_stereo_feature
            
            # Get the indices of the atoms involved in the bond
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Add the indices and features to the respective lists
            edges_index_list.extend([[i, j], [j, i]])
            edge_attr_list.extend([combined_features, combined_features])
        edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float64)
        edges_index_tensor = torch.tensor(edges_index_list, dtype=torch.long).t().contiguous()
    else:  
        edges_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, num_bond_features), dtype=torch.float64)

    # Convert the lists to tensors and return
    return edge_attr_tensor, edges_index_tensor

if __name__ == '__main__':
    data = pd.read_csv(root_dir / 'data/testcase_featurize.csv')
    smile = data['SMILES'][0]
    mol = get_mol(smile)
    edge_attr_list, edges_index_list = edge_feature(mol)
    print(f'Edge index shape: {edges_index_list.shape}')
    print(f'Edge index: {edges_index_list}')
    print(f'Edge attributes shape: {edge_attr_list.shape}')
    print(f'Edge attributes: {edge_attr_list}')
