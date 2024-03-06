import unittest
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from rdkit import Chem
from KGGraph.Chemistry.features import (
    get_symbol, get_atomic_number, get_period, get_group, get_atomicweight, get_num_valence_e, 
    get_chemical_group_block, get_hybridization, get_cip_code, is_chiral_center,
    get_formal_charge, get_total_num_hs, get_total_valence, get_num_radical_electrons,
    get_degree, is_aromatic, is_hetero, is_hydrogen_donor, is_hydrogen_acceptor,
    get_ring_size, is_in_ring, get_ring_membership_count, is_in_aromatic_ring, get_electronegativity,
    get_bond_type, is_conjugated, is_rotatable, get_stereo, get_bond_polarity, is_bond_in_ring,
    ELECTRONEGATIVITY
)

class TestMolecularFeatures(unittest.TestCase):

    def setUp(self):
        self.molecule = Chem.MolFromSmiles('CCO')  # ethanol as a test case
        self.atom = self.molecule.GetAtomWithIdx(1)  # Carbon atom
        self.bond = self.molecule.GetBondWithIdx(0)  # Bond between two carbon atoms

    # Atom tests
    def test_get_symbol(self):
        self.assertEqual(get_symbol(self.atom), 'C')
        
    def test_get_period(self):
        self.assertEqual(get_period(self.atom), 2)
        
    def test_get_group(self):
        self.assertEqual(get_group(self.atom), 14)
        
    def test_get_atomicweight(self):
        self.assertEqual(get_atomicweight(self.atom), 12.011)
        
    def test_get_num_valence_e(self):
        self.assertEqual(get_num_valence_e(self.atom), 4)
        
    def test_get_chemical_group_block(self):
        self.assertEqual(get_chemical_group_block(self.atom), 'Nonmetal')
    
    def test_get_atomic_number(self):
        self.assertEqual(get_atomic_number(self.atom), 6)

    def test_get_hybridization(self):
        self.assertEqual(get_hybridization(self.atom), 'SP3')

    def test_get_cip_code(self):
        self.assertIsNone(get_cip_code(self.atom))

    def test_is_chiral_center(self):
        # This molecule doesn't have chiral centers
        self.assertFalse(is_chiral_center(self.atom))

    def test_get_formal_charge(self):
        self.assertEqual(get_formal_charge(self.atom), 0)

    def test_get_total_num_hs(self):
        self.assertEqual(get_total_num_hs(self.atom), 2)

    def test_get_total_valence(self):
        self.assertEqual(get_total_valence(self.atom), 4)

    def test_get_num_radical_electrons(self):
        self.assertEqual(get_num_radical_electrons(self.atom), 0)

    def test_get_degree(self):
        self.assertEqual(get_degree(self.atom), 2)

    def test_is_aromatic(self):
        self.assertFalse(is_aromatic(self.atom))

    def test_is_hetero(self):
        self.assertFalse(is_hetero(self.atom))

    def test_is_hydrogen_donor(self):
        self.assertFalse(is_hydrogen_donor(self.atom))

    def test_is_hydrogen_acceptor(self):
        self.assertFalse(is_hydrogen_acceptor(self.atom))

    def test_get_ring_size(self):
        # Atom is not a part of ring
        self.assertEqual(get_ring_size(self.atom), 0)

    def test_is_in_ring(self):
        self.assertFalse(is_in_ring(self.atom))

    def test_get_ring_membership_count(self):
        self.assertEqual(get_ring_membership_count(self.atom), 0)

    def test_is_in_aromatic_ring(self):
        self.assertFalse(is_in_aromatic_ring(self.atom))


    def test_get_electronegativity_known_symbol(self):
        """Test getting electronegativity for an atom with a known symbol."""
        # Create an RDKit atom with a known symbol, e.g., 'C' (carbon)
        atom = Chem.Atom('C')
        # Retrieve the electronegativity using the function
        electronegativity = get_electronegativity(atom)
        # Get the expected electronegativity from the table
        expected_electronegativity = ELECTRONEGATIVITY['C']
        # Assert that the retrieved electronegativity matches the expected value
        self.assertEqual(electronegativity, expected_electronegativity)

    def test_get_electronegativity_unknown_symbol(self):
        """Test getting electronegativity for an atom with an unknown symbol."""
        atom = Chem.Atom('U') # not add Uranium
        # Retrieve the electronegativity using the function
        electronegativity = get_electronegativity(atom)
        # Assert that the retrieved electronegativity is 0.0 (default value for unknown symbols)
        self.assertEqual(electronegativity, 0.0)

    # Bond tests
    def test_get_bond_type(self):
        self.assertEqual(get_bond_type(self.bond), 'SINGLE')

    def test_is_conjugated(self):
        self.assertFalse(is_conjugated(self.bond))

    # def test_is_rotatable(self):
    #     # The bond in ethanol between two carbons should be rotatable
    #     self.assertTrue(is_rotatable(self.bond)) #TODO: check logic code

    def test_get_stereo(self):
        self.assertEqual(get_stereo(self.bond), 'STEREONONE')

    def test_get_bond_polarity(self):
        # Specific value depends on the implementation details
        polarity = get_bond_polarity(self.bond)
        self.assertTrue(isinstance(polarity, float))

    def test_is_bond_in_ring(self):
        self.assertFalse(is_bond_in_ring(self.bond))

if __name__ == '__main__':
    unittest.main()
