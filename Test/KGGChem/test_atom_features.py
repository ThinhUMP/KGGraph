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
class TestChemicalProperties(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.MolFromSmiles('CC(CC)OC1=CC=CC=C1')
        self.atoms = self.mol.GetAtoms()
        self.carbon = self.atoms[5]
        self.oxygen = self.atoms[4]

    def test_get_symbol(self):
        self.assertEqual(get_symbol(self.carbon), 'C')
        self.assertEqual(get_symbol(self.oxygen), 'O')

    def test_get_atomic_number(self):
        mol = Chem.MolFromSmiles('*CC')
        
        self.assertEqual(get_atomic_number(self.carbon), 6)
        self.assertEqual(get_atomic_number(mol.GetAtomWithIdx(0)), 0)

    def test_get_num_valence_e(self):
        self.assertEqual(get_num_valence_e(self.carbon), 4)
        self.assertEqual(get_num_valence_e(self.oxygen), 6)

    def test_get_chemical_group_block(self):
        self.assertEqual(get_chemical_group_block(self.carbon), [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])

    def test_get_hybridization(self):
        self.assertEqual(get_hybridization(self.carbon), 'SP2')
        self.assertEqual(get_hybridization(self.oxygen), 'SP2')

    def test_get_cip_code(self):
        self.assertIsNone(get_cip_code(self.carbon))
        
        self.carbon.SetProp("_CIPCode", "Carbon")
        self.assertEqual(get_cip_code(self.carbon), "Carbon")

    def test_is_chiral_center(self):
        self.assertFalse(is_chiral_center(self.carbon))
        self.assertTrue(is_chiral_center(self.atoms[1]))

    def test_get_formal_charge(self):
        self.assertEqual(get_formal_charge(self.carbon), 0)
        self.assertEqual(get_formal_charge(self.oxygen), 0)

    def test_get_total_num_hs(self):
        self.assertEqual(get_total_num_hs(self.oxygen), 0)
        self.assertEqual(get_total_num_hs(self.atoms[7]), 1) # Carbon with 1 H

    def test_get_total_valence(self):
        self.assertEqual(get_total_valence(self.carbon), 4)
        self.assertEqual(get_total_valence(self.oxygen), 2)

    def test_get_num_radical_electrons(self):
        self.assertEqual(get_num_radical_electrons(self.carbon), 0)
        self.assertEqual(get_num_radical_electrons(self.oxygen), 0)

    def test_get_degree(self):
        self.assertEqual(get_degree(self.carbon), 3)
        self.assertEqual(get_degree(self.oxygen), 2)

    def test_is_aromatic(self):
        self.assertTrue(is_aromatic(self.carbon))
        self.assertFalse(is_aromatic(self.oxygen))

    def test_is_hetero(self):
        self.assertFalse(is_hetero(self.carbon))
        self.assertTrue(is_hetero(self.oxygen))

    def test_is_hydrogen_donor(self):
        self.assertFalse(is_hydrogen_donor(self.carbon))
        self.assertFalse(is_hydrogen_donor(self.oxygen))

    def test_is_hydrogen_acceptor(self):
        self.assertFalse(is_hydrogen_acceptor(self.carbon))
        self.assertTrue(is_hydrogen_acceptor(self.oxygen))

    def test_get_ring_size(self):
        self.assertEqual(get_ring_size(self.carbon), 6)

    def test_is_in_ring(self):
        self.assertTrue(is_in_ring(self.carbon))
        self.assertFalse(is_in_ring(self.oxygen))

    def test_get_ring_membership_count(self):
        self.assertEqual(get_ring_membership_count(self.carbon), 1)

    def test_is_in_aromatic_ring(self):
        self.assertTrue(is_in_aromatic_ring(self.carbon))

    def test_get_electronegativity(self):
        self.assertEqual(get_electronegativity(self.carbon), 2.55)
        self.assertEqual(get_electronegativity(self.oxygen), 3.44)

if __name__ == '__main__':
    unittest.main()
