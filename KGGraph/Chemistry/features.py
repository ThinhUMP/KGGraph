from typing import Union
from rdkit import Chem
from rdkit.Chem import Lipinski, Crippen, rdMolDescriptors, rdPartialCharges
from mendeleev import element
import numpy as np
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.chemutils import *
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': 0.0,
    'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Ne': 0.0, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'Ar': 0.0, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,
    'Se': 2.55, 'Br': 2.96, 'Kr': 3.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6,
    'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78,
    'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6,
    # Continue adding more
}


### Atom type
def get_symbol(atom: Chem.Atom) -> str:
    """Get the symbol of the atom."""
    return atom.GetSymbol()

def get_period(atom: Chem.Atom) -> int:
    """Get the period of the atom."""
    atom_mendeleev = element(atom.GetSymbol())
    return atom_mendeleev.period

def get_group(atom: Chem.Atom) -> str:
    """Get the period of the atom."""
    atom_mendeleev = element(atom.GetSymbol())
    return atom_mendeleev.group

def get_hybridization(atom: Chem.Atom) -> str:
    """Get the hybridization type of the atom."""
    return atom.GetHybridization().name

def get_cip_code(atom: Chem.Atom) -> Union[None, str]:
    """Get the CIP code of the atom, if available."""
    if atom.HasProp("_CIPCode"):
        return atom.GetProp("_CIPCode")
    return None

def is_chiral_center(atom: Chem.Atom) -> bool:
    """Determine if the atom is a chiral center."""
    return atom.HasProp("_ChiralityPossible")

def get_formal_charge(atom: Chem.Atom) -> int:
    """Get the formal charge of the atom."""
    return atom.GetFormalCharge()

def get_total_num_hs(atom: Chem.Atom) -> int:
    """Get the total number of hydrogen atoms connected to the atom."""
    return atom.GetTotalNumHs()

def get_total_valence(atom: Chem.Atom) -> int:
    """Get the total valence of the atom."""
    return atom.GetTotalValence()

def get_num_radical_electrons(atom: Chem.Atom) -> int:
    """Get the number of radical electrons of the atom."""
    return atom.GetNumRadicalElectrons()

def get_degree(atom: Chem.Atom) -> int:
    """Get the degree of the atom (number of bonded neighbors)."""
    return atom.GetDegree()

def is_aromatic(atom: Chem.Atom) -> bool:
    """Check if the atom is part of an aromatic system."""
    return atom.GetIsAromatic()

def is_hetero(atom: Chem.Atom) -> bool:
    """Check if the atom is a heteroatom."""
    mol = atom.GetOwningMol()
    return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]

def is_hydrogen_donor(atom: Chem.Atom) -> bool:
    """Check if the atom is a hydrogen bond donor."""
    mol = atom.GetOwningMol()
    return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]

def is_hydrogen_acceptor(atom: Chem.Atom) -> bool:
    """Check if the atom is a hydrogen bond acceptor."""
    mol = atom.GetOwningMol()
    return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]

def get_ring_size(atom: Chem.Atom) -> int:
    """Get the ring size for the smallest ring the atom is a part of."""
    size = 0
    if atom.IsInRing():
        while not atom.IsInRingSize(size):
            size += 1
    return size

def is_in_ring(atom: Chem.Atom) -> bool:
    """Check if the atom is part of any ring."""
    return atom.IsInRing()

def get_ring_membership_count(atom: Chem.Atom) -> int:
    """Get the number of rings the atom is a part of."""
    mol = atom.GetOwningMol()
    ring_info = mol.GetRingInfo()
    return len([ring for ring in ring_info.AtomRings() if atom.GetIdx() in ring])


def is_in_aromatic_ring(atom: Chem.Atom) -> bool:
    """Check if the atom is part of an aromatic ring."""
    return atom.GetIsAromatic()

def get_electronegativity(atom: Chem.Atom) -> float:
    """Get the electronegativity of the atom from the ELECTRONEGATIVITY table."""
    # Retrieve the atom's symbol
    symbol = atom.GetSymbol()
    # Get the electronegativity value from the table, return 0.0 if not found
    return ELECTRONEGATIVITY.get(symbol, 0.0)


#### Bond type

def get_bond_type(bond: Chem.Bond) -> str:
    """Get the type of the bond."""
    return bond.GetBondType().name

def is_conjugated(bond: Chem.Bond) -> bool:
    """Check if the bond is conjugated."""
    return bond.GetIsConjugated()

def is_rotatable(bond: Chem.Bond) -> bool: 
    """Check if the bond is rotatable."""
    mol = bond.GetOwningMol()
    atom_indices = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return atom_indices in Lipinski._RotatableBonds(mol)

def get_stereo(bond: Chem.Bond) -> str:
    """Get the stereochemistry of the bond."""
    return bond.GetStereo().name

def get_bond_polarity(bond: Chem.Bond) -> float:
    """Estimate the polarity of the bond based on the electronegativity difference."""
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    
    # Get electronegativity values from the lookup table
    en1 = ELECTRONEGATIVITY.get(atom1.GetSymbol())
    en2 = ELECTRONEGATIVITY.get(atom2.GetSymbol())
    
    # Ensure that we have valid electronegativity values for both atoms
    if en1 is not None and en2 is not None:
        return abs(en1 - en2)
    else:
        # Handle cases where electronegativity is not defined (e.g., for a missing element in the table)
        return 0.0  # or you might want to raise an exception or return None

def is_bond_in_ring(bond: Chem.Bond) -> bool:
    """Check if the bond is part of a ring."""
    return bond.IsInRing()