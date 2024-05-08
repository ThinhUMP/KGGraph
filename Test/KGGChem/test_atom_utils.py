import unittest
from rdkit import Chem
import numpy as np

import sys
import pathlib

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGChem.atom_utils import (
    get_mol,
    get_smiles,
    sanitize,
    get_atomic_number,
    atomic_num_features,
    set_atommap,
    atom_equal,
    copy_atom,
    get_atom_types,
)


class TestChemUtils(unittest.TestCase):

    def setUp(self):
        self.smile = "CO"
        self.mol = Chem.MolFromSmiles(self.smile)

    def tearDown(self):
        pass

    def test_get_mol(self):
        """Test molecule generation from SMILES."""
        self.assertIsInstance(get_mol(self.smile), Chem.Mol)
        self.assertIsNone(get_mol("C=C#C"))

    def test_get_smiles(self):
        """Test SMILES generation from molecule."""
        self.assertEqual(get_smiles(self.mol), self.smile)

    def test_sanitize(self):
        """Test molecule sanitization and optional kekulization."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        sanitized = sanitize(mol, kekulize=True)
        self.assertIsInstance(sanitized, Chem.Mol)
        self.assertIsNone(sanitize(Chem.MolFromSmiles("c1cccc1")))

    def test_get_atomic_number(self):
        """Test atomic number retrieval."""
        self.assertEqual(get_atomic_number(self.mol.GetAtomWithIdx(0)), 6)  # Carbon
        self.assertEqual(get_atomic_number(self.mol.GetAtomWithIdx(1)), 8)  # Oxygen

    def test_get_atom_types(self):
        """Test atomic number retrieval for a list of molecules."""
        smile_list = [self.smile, "CCN"]
        self.assertEqual(get_atom_types(smile_list), [6, 7, 8])

    def test_atomic_num_features(self):
        """Test atomic number features for a molecule."""
        atom_types = [6, 7, 8]
        atomic_num_feature = atomic_num_features(self.mol, atom_types)
        target = np.array([[1, 0, 0], [0, 0, 1]])
        self.assertEqual(atomic_num_feature.all(), target.all())

    def test_set_atommap(self):
        """Test setting atom map numbers for all atoms in a molecule."""
        modified_mol = set_atommap(self.mol, num=2)
        for atom in modified_mol.GetAtoms():
            self.assertEqual(atom.GetAtomMapNum(), 2)

    def test_atom_equal(self):
        """Test atom equality based on symbol and formal charge."""
        self.assertTrue(
            atom_equal(self.mol.GetAtomWithIdx(0), self.mol.GetAtomWithIdx(0))
        )  # Carbon with itself
        self.assertFalse(
            atom_equal(self.mol.GetAtomWithIdx(0), self.mol.GetAtomWithIdx(1))
        )  # Carbon with Oxygen

    def test_copy_atom(self):
        """Test copying an atom optionally preserving its atom map number."""
        original_atom = self.mol.GetAtomWithIdx(0)
        copied_atom = copy_atom(original_atom)
        self.assertTrue(atom_equal(original_atom, copied_atom))
        self.assertEqual(copied_atom.GetAtomMapNum(), original_atom.GetAtomMapNum())


if __name__ == "__main__":
    unittest.main()
