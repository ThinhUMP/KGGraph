import unittest
from rdkit import Chem
from KGGraph.KGGChem.gasteiger_adj import GasteigerADJ  
import numpy as np

class TestGasteigerADJ(unittest.TestCase):

    def setUp(self):
        # Create a simple molecule for testing
        self.smiles = 'CCO'  # Ethanol
        self.mol = Chem.MolFromSmiles(self.smiles)

    def test_add_atom_mapping(self):
        mapped_mol = GasteigerADJ.add_atom_mapping(self.mol)
        # Check if atom mapping numbers are correctly assigned
        for idx, atom in enumerate(mapped_mol.GetAtoms()):
            self.assertEqual(atom.GetAtomMapNum(), idx + 1)

    def test_renumber_and_calculate_charges(self):
        mapped_mol, charges = GasteigerADJ.renumber_and_calculate_charges(self.smiles)
        # Check if the molecule is correctly mapped
        self.assertEqual(mapped_mol.GetAtomWithIdx(0).GetAtomMapNum(), 1)
        # Check if charges dictionary is populated when calculate_charges is True
        self.assertTrue(len(charges) > 0)

    def test_calculate_directed_adjacency_matrix(self):
        _, charges = GasteigerADJ.renumber_and_calculate_charges(self.smiles, calculate_charges=True)
        mapped_mol = GasteigerADJ.add_atom_mapping(self.mol)
        adj_matrix = GasteigerADJ.calculate_directed_adjacency_matrix(mapped_mol, charges)
        # Basic validation of the adjacency matrix
        self.assertIsInstance(adj_matrix, np.ndarray)
        self.assertEqual(adj_matrix.shape, (mapped_mol.GetNumAtoms(), mapped_mol.GetNumAtoms()))
        
if __name__ == '__main__':
    unittest.main()
