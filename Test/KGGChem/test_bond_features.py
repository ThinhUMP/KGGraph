import unittest
from rdkit import Chem
import sys
import pathlib
from KGGraph.KGGChem.bond_features import (
    get_bond_type,
    is_conjugated,
    is_rotatable,
    get_stereo,
    get_bond_polarity,
    is_bond_in_ring,
    bond_type_feature,
)

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)


class TestBondProperties(unittest.TestCase):
    def setUp(self):
        # Create a molecule that includes various types of bonds
        self.mol = Chem.MolFromSmiles(
            "O=C(C)C=C"
        )  # Contains single, double, and rotatable bonds
        self.bonds = list(self.mol.GetBonds())
        self.double_bond = self.bonds[0]  # Double bond C=O
        self.single_bond = self.bonds[1]  # Single bond C-C

    def tearDown(self):
        pass

    def test_get_bond_type(self):
        self.assertEqual(get_bond_type(self.double_bond), "DOUBLE")
        self.assertEqual(get_bond_type(self.single_bond), "SINGLE")

    def test_is_conjugated(self):
        self.assertTrue(is_conjugated(self.double_bond))
        self.assertFalse(is_conjugated(self.single_bond))

    def test_is_rotatable(self):
        # Third bond in the molecule should be rotatable
        self.assertTrue(is_rotatable(self.bonds[2]))  # Bond between C and C
        self.assertFalse(is_rotatable(self.double_bond))

    def test_get_stereo(self):
        self.assertEqual(get_stereo(self.single_bond), "STEREONONE")

    def test_get_bond_polarity(self):
        self.assertEqual(get_bond_polarity(self.bonds[2]), 0)  # Bond between C and C
        self.assertGreater(get_bond_polarity(self.double_bond), 0)

    def test_is_bond_in_ring(self):
        self.assertFalse(
            is_bond_in_ring(self.single_bond)
        )  # No rings in the test molecule

    def test_bond_type_feature(self):
        bonds = Chem.MolFromSmiles("c1ccccc1").GetBonds()
        self.assertEqual(
            bond_type_feature(bonds[0]), [1, 0.5, 1, 1]
        )  # Example vector for 'AROMATIC'
        # Test if the function returns correct feature vector
        self.assertEqual(
            bond_type_feature(self.double_bond), [1, 1, 1, 0]
        )  # Example vector for 'DOUBLE' conjugated
        self.assertEqual(
            bond_type_feature(self.single_bond), [1, 0, 0, 0]
        )  # Example vector for 'SINGLE' NOT conjugated
        self.assertEqual(
            bond_type_feature(self.bonds[2]), [1, 0, 1, 0]
        )  # Example vector for 'SINGLE' conjugated
        self.assertEqual(
            bond_type_feature(self.bonds[3]), [1, 1, 1, 0]
        )  # Example vector for 'DOUBLE' conjugated


if __name__ == "__main__":
    unittest.main()
