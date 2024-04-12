from rdkit import Chem
from .atom_utils import atom_equal

def bond_match(mol1: Chem.Mol, a1: int, b1: int, mol2: Chem.Mol, a2: int, b2: int) -> bool:
    """
    Check if bonds in two molecules match based on their connecting atoms.

    Parameters:
    - mol1 (Chem.Mol): The first molecule.
    - a1 (int): Atom index in the first molecule.
    - b1 (int): Another atom index in the first molecule.
    - mol2 (Chem.Mol): The second molecule.
    - a2 (int): Atom index in the second molecule.
    - b2 (int): Another atom index in the second molecule.

    Returns:
    - bool: True if the bonds match, False otherwise.
    """
    a1,b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2,b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1,a2) and atom_equal(b1,b2)