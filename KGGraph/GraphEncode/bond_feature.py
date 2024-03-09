import torch
import numpy as np
from tqdm import tqdm
import time
from typing import List
import sys
import pathlib
import json
from rdkit import Chem
from rdkit.Chem import Mol
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.chemutils import get_mol
from KGGraph.Chemistry.features import (
    get_bond_type, is_conjugated, is_rotatable, get_stereo, get_bond_polarity, is_bond_in_ring
)

# Load bond dictionaries only once to avoid repeated file access.
with open('./data/bond_dict.json', 'r') as f:
    bond_dict = json.load(f)

with open('./data/bond_stereo_dict.json', 'r') as f:
    bond_stereo_dict = json.load(f)

class EdgeFeature:
    """
    This class computes various bond features for a dataset of molecules.
    """

    def __init__(self, mol: Chem.Mol):
        """
        Initialize the class with molecule data and the column containing SMILES representations.

        Parameters:
        - data: DataFrame containing the dataset.
        - smile_col: String name of the column containing the SMILES data.
        """
        self.mol = mol

    def get_feature(self):
        """
        Extract bond features for each molecule in the dataset.

        Returns:
        A list of tensors, each representing bond features for a molecule.
        """
        bond_features = [self.compute_features(bond) for bond in self.mol.GetBonds()]
        return bond_features

    def compute_features(self, bond):
        """
        Calculate features for a single bond.

        Parameters:
        - bond: The bond object.

        Returns:
        A tensor containing the bond's features.
        """
        features = [
            is_conjugated(bond),
            is_rotatable(bond),
            get_bond_polarity(bond),
            is_bond_in_ring(bond)
        ]
        return torch.tensor(features, dtype=torch.float64)

    def edge_type(self):
        """
        Get one-hot encoded vectors representing the types of bonds in each molecule.

        Returns:
        A list of tensors with one-hot encoded bond types.
        """
        bond_mol = [torch.tensor(bond_dict.get(get_bond_type(bond), None), dtype=torch.float64) for bond in self.mol.GetBonds()]
        return bond_mol

    def edge_stereo(self):
        """
        Get one-hot encoded vectors representing the stereochemistry of bonds in each molecule.

        Returns:
        A list of tensors with one-hot encoded bond stereochemistry.
        """
        stereo_mol = [torch.tensor(bond_stereo_dict.get(get_stereo(bond), None), dtype=torch.float64) for bond in self.mol.GetBonds()]
        return stereo_mol
    
    def get_edge_index(self):
        edges_list = []
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges_list.append([i, j], [j, i])
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.float)
        return edge_index

    def feature(self):
        """
        Combine all calculated features for each bond in each molecule.

        Returns:
        A list of tensors, each representing the combined features for a molecule.
        """
        start_time = time.time()

        # Get individual feature components
        feature_mol = self.get_feature()
        bond_mol = self.edge_type()
        stereo_mol = self.edge_stereo()

        # Combine features for each molecule

        for i in range(len(mol.GetBonds())):
            edge_attr = torch.cat((feature_mol[i], bond_mol[i], stereo_mol[i]), dim=0)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return edge_attr
    
if __name__ == '__main__':
    import pandas as pd
    import sys
    import pathlib
    root_dir = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(root_dir)
    data = pd.read_csv('data/testcase_featurize.csv')
    smile = data['SMILES'][0]
    mol = get_mol(smile)
    edge_feature = EdgeFeature(mol)
    edge_attr = edge_feature.feature()
    print(edge_attr.shape)
    print(edge_attr)