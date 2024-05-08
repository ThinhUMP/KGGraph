import unittest
from rdkit import Chem
import sys
import pathlib
from KGGraph.KGGChem.bond_utils import bond_match

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)


class TestBondMatch(unittest.TestCase):
    def setUp(self):
        # Example molecules
        self.mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        self.mol2 = Chem.MolFromSmiles("CCO")  # Another ethanol molecule
        self.mol3 = Chem.MolFromSmiles("CCN")  # Ethylamine

    def test_bond_match_true(self):
        # Test matching bond (C-O in both molecules)
        self.assertTrue(bond_match(self.mol1, 1, 2, self.mol2, 1, 2))
        # Test non-matching bond due to different connecting atoms (C-O vs C-N)
        self.assertFalse(bond_match(self.mol1, 1, 2, self.mol3, 1, 2))
        # Test non-matching bond due to different order of atoms
        self.assertFalse(bond_match(self.mol1, 2, 1, self.mol2, 1, 2))
        # Test non-matching bond due to completely different molecules
        self.assertFalse(bond_match(self.mol1, 1, 2, self.mol3, 1, 2))


if __name__ == "__main__":
    unittest.main()
