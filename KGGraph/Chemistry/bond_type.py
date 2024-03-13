from rdkit import Chem
import json
import sys
import pathlib
from mendeleev import element
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.features import get_formal_charge, get_bond_type
    
with open('./data/bond_dict.json', 'r') as f:
    bond_dict = json.load(f)

class GetBondTypeFeature:
    """
    A class to determine bond type features based on the bond and its connected atoms.
    """

    @staticmethod
    def is_conjugated_atom(atom):
        """
        Determine if an atom could be a part of conjugated system (Ex: C=CN, N is conjugated to pi).

        Parameters:
        atom (rdkit.Chem.rdchem.Atom): The atom to check.

        Returns:
        bool: True if the atom is N, O, or S and has a negative or neutral formal charge.
        """
        return atom.GetSymbol() in ['N', 'O', 'S'] and get_formal_charge(atom) <= 0

    @staticmethod
    def has_multiple_bond(atom):
        """
        Check if an atom is involved in a multiple bond (double or triple).

        Parameters:
        atom (rdkit.Chem.rdchem.Atom): The atom to check.

        Returns:
        bool: True if the atom has a double or triple bond, False otherwise.
        """
        return any(b.GetBondType() in (Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE) for b in atom.GetBonds())

    @staticmethod
    def feature(bond):
        """
        Determine the feature of a bond based on its type and the atoms it connects.

        Parameters:
        bond (rdkit.Chem.rdchem.Bond): The bond to analyze.

        Returns:
        list: A list representing the bond type feature.
        """
        # Directly return the feature if the bond is not single
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            return bond_dict.get(get_bond_type(bond), [0] * (len(bond_dict) - 1) + [1])
        else: # If the bond is single
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Determine if the bond is conjugated based on the connected atoms. Ex: C=C-N or N-C=C
            if (GetBondTypeFeature.is_conjugated_atom(begin_atom) and GetBondTypeFeature.has_multiple_bond(end_atom)) or \
            (GetBondTypeFeature.is_conjugated_atom(end_atom) and GetBondTypeFeature.has_multiple_bond(begin_atom)):
                bond_type_feature = bond_dict.get('CONJUGATE')
            else:
                # Additional check for multiple bonds to determine if the bond is conjugated. Ex: C=C-C=C
                bond_type_feature = bond_dict.get('CONJUGATE') if GetBondTypeFeature.has_multiple_bond(begin_atom) and GetBondTypeFeature.has_multiple_bond(end_atom) else bond_dict.get('SINGLE')

        return bond_type_feature