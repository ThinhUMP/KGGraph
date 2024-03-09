import torch
import numpy as np
from tqdm import tqdm
import time
from typing import List
import sys
import pathlib
import json
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

class BondFeature:
    """
    This class computes various bond features for a dataset of molecules.
    """

    def __init__(self, data, smile_col: str):
        """
        Initialize the class with molecule data and the column containing SMILES representations.

        Parameters:
        - data: DataFrame containing the dataset.
        - smile_col: String name of the column containing the SMILES data.
        """
        self.data = data
        self.smile_col = smile_col
        self.smiles = data[smile_col].tolist()
        self.molecules = [get_mol(smile) for smile in self.smiles]

    def get_feature(self):
        """
        Extract bond features for each molecule in the dataset.

        Returns:
        A list of tensors, each representing bond features for a molecule.
        """
        feature_mols = []
        for molecule in tqdm(self.molecules):
            bond_features = [self.compute_features(bond) for bond in molecule.GetBonds()]
            feature_mols.append(torch.stack(bond_features))
        return feature_mols

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

    def bond_type(self):
        """
        Get one-hot encoded vectors representing the types of bonds in each molecule.

        Returns:
        A list of tensors with one-hot encoded bond types.
        """
        bond_mols = []
        for molecule in self.molecules:
            bond_mol = [torch.tensor(bond_dict.get(get_bond_type(bond), None), dtype=torch.float64) for bond in molecule.GetBonds()]
            bond_mols.append(torch.stack(bond_mol))
        return bond_mols

    def bond_stereo(self):
        """
        Get one-hot encoded vectors representing the stereochemistry of bonds in each molecule.

        Returns:
        A list of tensors with one-hot encoded bond stereochemistry.
        """
        stereo_mols = []
        for molecule in self.molecules:
            stereo_mol = [torch.tensor(bond_stereo_dict.get(get_stereo(bond), None), dtype=torch.float64) for bond in molecule.GetBonds()]
            stereo_mols.append(torch.stack(stereo_mol))
        return stereo_mols

    def feature(self):
        """
        Combine all calculated features for each bond in each molecule.

        Returns:
        A list of tensors, each representing the combined features for a molecule.
        """
        start_time = time.time()

        # Get individual feature components
        feature_mols = self.get_feature()
        bond_mols = self.bond_type()
        stereo_mols = self.bond_stereo()

        # Combine features for each molecule
        combined_features = []
        for i in range(len(self.smiles)):
            combined = torch.cat((feature_mols[i], bond_mols[i], stereo_mols[i]), dim=1)
            combined_features.append(combined)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return combined_features
    
if __name__ == '__main__':
    import pandas as pd
    import sys
    import pathlib
    root_dir = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(root_dir)
    data = pd.read_csv('data/testcase_featurize.csv')
    atom_feature_obj = BondFeature(data=data, smile_col='SMILES')
    atom_features = atom_feature_obj.feature()
    print(atom_features[0].shape)
    print(len(atom_features))