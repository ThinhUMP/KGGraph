import unittest
from rdkit import Chem

import sys
import pathlib

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGChem.hybridization import HybridizationFeaturize


class TestHybridizationFeaturize(unittest.TestCase):

    def setUp(self):
        # Create some molecules to test with
        self.mol_ethane = Chem.MolFromSmiles("CC")
        self.mol_ethene = Chem.MolFromSmiles("C=C")
        self.mol_ethyne = Chem.MolFromSmiles("C#C")

        # Get atoms from molecules for testing
        self.atom_ethane = self.mol_ethane.GetAtomWithIdx(0)  # Carbon atom in ethane
        self.atom_ethene = self.mol_ethene.GetAtomWithIdx(0)  # Carbon atom in ethene
        self.atom_ethyne = self.mol_ethyne.GetAtomWithIdx(0)  # Carbon atom in ethyne

    def test_total_sigma_bond(self):
        self.assertEqual(
            HybridizationFeaturize.total_sigma_bond(self.atom_ethane), 4
        )  # 3 single bonds in ethane
        self.assertEqual(
            HybridizationFeaturize.total_sigma_bond(self.atom_ethene), 3
        )  # 2 single bonds in ethene
        self.assertEqual(
            HybridizationFeaturize.total_sigma_bond(self.atom_ethyne), 2
        )  # 1 single bond in ethyne

    def test_num_bond_hybridization(self):
        # Assuming get_hybridization returns 'SP3' for ethane, 'SP2' for ethene, and 'SP' for ethyne
        self.assertEqual(
            HybridizationFeaturize.num_bond_hybridization(self.atom_ethane), 4
        )  # SP3 hybridization
        self.assertEqual(
            HybridizationFeaturize.num_bond_hybridization(self.atom_ethene), 3
        )  # SP2 hybridization
        self.assertEqual(
            HybridizationFeaturize.num_bond_hybridization(self.atom_ethyne), 2
        )  # SP hybridization

    def test_feature(self):
        # Example test for the 'feature' method
        ethane_features = HybridizationFeaturize.feature(self.atom_ethane)

        self.assertEqual(ethane_features[0], 4)  # Total single bonds
        self.assertEqual(
            ethane_features[1], 0
        )  # Number of lone pairs, assuming 0 for simplicity
        self.assertEqual(
            ethane_features[2], [1, 3, 0, 4, 0]
        )  # Feature vector for ethane

        # Add more tests for different atoms to thoroughly test the 'feature' method


if __name__ == "__main__":
    unittest.main()
