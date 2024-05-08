import unittest
from rdkit import Chem
import sys
import pathlib
from KGGraph.KGGChem.chemutils import (
    copy_edit_mol,
    get_clique_mol,
)

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)


class TestMolecularFunctions(unittest.TestCase):

    def setUp(self):
        # Simple molecules for testing
        self.mol = Chem.MolFromSmiles("CC1=CC=CC=C1")  # Toluene

    def test_copy_edit_mol(self):
        # Test if copy is identical to original
        copy_mol = copy_edit_mol(self.mol)
        self.assertEqual(Chem.MolToSmiles(copy_mol), Chem.MolToSmiles(self.mol))

    def test_get_clique_mol(self):
        # Test generating a substructure
        clique_mol = get_clique_mol(self.mol, [1, 2, 3, 4, 5, 6])
        self.assertTrue(clique_mol is not None)
        self.assertEqual(Chem.MolToSmiles(clique_mol), "c1ccccc1")


if __name__ == "__main__":
    unittest.main()
