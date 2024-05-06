from .atom_utils import get_smiles
from rdkit import Chem
from rdkit.Chem import Lipinski
from .atom_features import ELECTRONEGATIVITY
from typing import List
import json
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
with open(root_dir+'/Data/feature/bond_dict.json', 'r') as f:
    bond_dict = json.load(f)

#### Bond type
def get_bond_type(bond: Chem.Bond) -> str:
    """Get the type of the bond."""
    return bond.GetBondType().name

def is_conjugated(bond: Chem.Bond) -> bool:
    """Check if the bond is conjugated."""
    try:
        return bond.GetIsConjugated()
    except:
        print(f"{get_smiles(bond.GetOwningMol())} contains {get_bond_type(bond)} which can not get is conjugated.")
        return False

def is_rotatable(bond: Chem.Bond) -> bool: 
    """Check if the bond is rotatable."""
    try:
        mol = bond.GetOwningMol()
        atom_indices = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        return atom_indices in Lipinski._RotatableBonds(mol)
    except:
        print(f"{get_smiles(bond.GetOwningMol())} contains {get_bond_type(bond)} which can not get is rotatable.")
        return False

def get_stereo(bond: Chem.Bond) -> str:
    """Get the stereochemistry of the bond."""
    return bond.GetStereo().name

def get_bond_polarity(bond: Chem.Bond) -> int:
    """Estimate the polarity of the bond based on the electronegativity difference."""
    try:
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
    except:
        print(f"{get_smiles(bond.GetOwningMol())} contains {get_bond_type(bond)} which can not get bond polarity.")
        return 0.0

def is_bond_in_ring(bond: Chem.Bond) -> bool:
    """Check if the bond is part of a ring."""
    try:
        return bond.IsInRing()
    except:
        print(f"{get_smiles(bond.GetOwningMol())} contains {get_bond_type(bond)} which can not get is in ring.")
        return False
    
def bond_type_feature(bond) -> List[int]:
    """Determine the feature representation of a bond in a molecular structure. The function 
    categorizes the bond into specific types, such as aromatic or conjugated, and returns 
    a corresponding feature vector from a predefined dictionary."""
    
    # Get the bond type as a string representation
    bond_type = get_bond_type(bond)
    
    # Check for conjugated bond type
    if is_conjugated(bond):
        bond_type_feature = bond_dict.get(bond_type, [0,0,0,0])
        bond_type_feature[2] = 1
        return bond_type_feature
    
    # Return the bond type feature or a default 'other' type feature vector
    return bond_dict.get(bond_type, [0,0,0,0])

# Auxiliary functions and dictionary must be defined:
# get_bond_type(bond): Should return a string representation of the bond type.
# is_conjugated(bond): Should return a boolean indicating if the bond is conjugated.
# bond_dict: A dictionary mapping bond type strings to feature lists.