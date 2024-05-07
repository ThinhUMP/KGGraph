import unittest

from rdkit import Chem

import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGChem.atom_features import (
    get_symbol, get_atomic_number, get_num_valence_e, get_chemical_group_block, get_hybridization, get_cip_code,
    is_chiral_center, get_formal_charge, get_total_num_hs, get_total_valence, get_num_radical_electrons, get_degree,
    is_aromatic, is_hetero, is_hydrogen_donor, is_hydrogen_acceptor, get_ring_size, is_in_ring, get_ring_membership_count,
    is_in_aromatic_ring, get_electronegativity
)

import unittest
from rdkit import Chem

class TestAtomFeatures(unittest.TestCase):
    """
    This class contains a suite of unit tests for various atom features
    extracted using the RDKit library, focusing on an aromatic molecule
    with an additional hydroxyl group.
    """
    
    def setUp(self):
        """
        Set up the testing environment before each test method is executed.
        A benzene ring with a hydroxyl group ('c1ccccc1(O)') is created and
        the first atom (carbon) and the first bond are accessed for testing.
        """
        self.mol = Chem.MolFromSmiles('c1ccccc1(O)')
        self.atom = self.mol.GetAtomWithIdx(0)
        
    def tearDown(self):
        """
        Clean up after each test method is executed.
        """
        pass
        
    def test_get_symbol(self):
        """
        Test the extraction of the chemical symbol of an atom.
        """
        self.assertEqual(get_symbol(self.atom), 'C')
    
    def test_get_atomic_number(self):
        """
        Test that the correct atomic number is returned for an atom.
        """
        self.assertEqual(get_atomic_number(self.atom), 6)
        
    def test_get_num_valence_e(self):
        """
        Test the number of valence electrons of an atom.
        """
        self.assertEqual(get_num_valence_e(self.atom), 4)
        
    def test_get_chemical_group_block(self):
        """
        Test the extraction of the atom's chemical group block as a one-hot encoded list.
        """
        self.assertEqual(get_chemical_group_block(self.atom), [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])
        
    def test_get_hybridization(self):
        """
        Test the hybridization state of the atom.
        """
        self.assertEqual(get_hybridization(self.atom), 'SP2')
        
    def test_get_cip_code(self):
        """
        Test the retrieval of the CIP stereochemistry code for the atom.
        """
        self.assertIsNone(get_cip_code(self.atom))
        
    def test_is_chiral_center(self):
        """
        Test whether the atom is a chiral center.
        """
        self.assertFalse(is_chiral_center(self.atom))
        
    def test_get_formal_charge(self):
        """
        Test the formal charge of the atom.
        """
        self.assertEqual(get_formal_charge(self.atom), 0)
        
    def test_get_total_num_hs(self):
        """
        Test the total number of hydrogen atoms attached to the atom.
        """
        self.assertEqual(get_total_num_hs(self.atom), 1)
        
    def test_get_total_valence(self):
        """
        Test the total valence of the atom.
        """
        self.assertEqual(get_total_valence(self.atom), 4)
        
    def test_get_num_radical_electrons(self):
        """
        Test the number of radical electrons on the atom.
        """
        self.assertEqual(get_num_radical_electrons(self.atom), 0)
        
    def test_get_degree(self):
        """
        Test the degree of the atom, i.e., the number of directly bonded neighbors.
        """
        self.assertEqual(get_degree(self.atom), 2)
        
    def test_is_aromatic(self):
        """
        Test whether the atom is part of an aromatic system.
        """
        self.assertTrue(is_aromatic(self.atom))
        
    def test_is_hetero(self):
        """
        Test whether the atom is a heteroatom. Tests both carbon (false) and oxygen (true).
        """
        self.assertFalse(is_hetero(self.atom))
        self.assertTrue(is_hetero(self.mol.GetAtomWithIdx(6)))
        
    def test_is_hydrogen_donor(self):
        """
        Test whether the atom can act as a hydrogen donor. Tests both carbon (false) and oxygen (true).
        """
        self.assertFalse(is_hydrogen_donor(self.atom))
        self.assertTrue(is_hydrogen_donor(self.mol.GetAtomWithIdx(6)))
        
    def test_is_hydrogen_acceptor(self):
        """
        Test whether the atom can act as a hydrogen acceptor.
        """
        self.assertFalse(is_hydrogen_acceptor(self.atom))
        
    def test_get_ring_size(self):
        """
        Test the size of the smallest ring the atom is a member of.
        """
        self.assertEqual(get_ring_size(self.atom), 6)
        
    def test_is_in_ring(self):
        """
        Test whether the atom is part of any ring structure.
        """
        self.assertTrue(is_in_ring(self.atom))
        
    def test_get_ring_membership_count(self):
        """
        Test the number of ring memberships of the atom.
        """
        self.assertEqual(get_ring_membership_count(self.atom), 1)
        
    def test_is_in_aromatic_ring(self):
        """
        Test whether the atom is part of an aromatic ring.
        """
        self.assertTrue(is_in_aromatic_ring(self.atom))

    def test_get_electronegativity(self):
        """
        Test the electronegativity of the atom.
        """
        self.assertEqual(get_electronegativity(self.atom), 2.55)

if __name__ == '__main__':
    unittest.main()