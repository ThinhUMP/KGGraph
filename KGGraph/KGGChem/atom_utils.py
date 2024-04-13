import rdkit
import rdkit.Chem as Chem
from typing import List, Tuple, Set, Optional
import numpy as np
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

idxfunc = lambda a: a.GetAtomMapNum() - 1

def get_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Generate a molecule object from a SMILES string.

    Parameters:
    - smiles (str): The SMILES string representing the molecule.

    Returns:
    - Optional[Chem.Mol]: The generated molecule object, or None if molecule generation fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    # if mol is not None: 
    #     Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol

def get_smiles(mol: Chem.Mol) -> str:
    """
    Convert a molecule object to a SMILES string.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - str: The SMILES representation of the molecule.
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=False)

def sanitize(mol: Chem.Mol, kekulize: bool = False) -> Optional[Chem.Mol]:
    """
    Sanitize the given molecule and optionally kekulize it.

    Parameters:
    - mol (Chem.Mol): The molecule to sanitize.
    - kekulize (bool): If True, kekulize the molecule.

    Returns:
    - Optional[Chem.Mol]: The sanitized molecule, or None if an error occurs.
    """
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def get_atomic_number(atom: Chem.Atom) -> int:
    """Get the atomic number of the atom."""
    try:
        atomic_number = atom.GetAtomicNum()
        if atomic_number is None:
            atomic_number = 0
    except:
        atomic_number = 0
        print(f"{get_smiles(atom.GetOwningMol())} contains atom which can not get atomic number.")
    return atomic_number

def get_atom_types(smiles: List[str]) -> List[int]:
    """
    Returns a list of unique atomic numbers present in the molecules represented by the given SMILES strings.
    
    Args:
        smiles (str): The list of SMILES strings representing the molecules.
        
    Returns:
        List[int]: A sorted list of unique atomic numbers present in the molecules.
    """
    mols = [get_mol(smile) for smile in smiles]
    atom_types = []
    for mol in mols:
        for atom in mol.GetAtoms():
            if get_atomic_number(atom) not in atom_types:
                atom_types.append(get_atomic_number(atom))
    atom_types.sort()
    return atom_types

# def atomic_num_features(mol, atom_types):
#     """_summary_

#     Args:
#         m (rdkit mol): Molecule to be transformed into a graph.

#         atom_types (list): List of all atom types present in the dataset 
#             represented by their atomic numbers.
#     """
#     atomic_features = np.zeros((mol.GetNumAtoms(), len(atom_types)))
#     for idx, atom in enumerate(mol.GetAtoms()):
#         atomic_features[idx] = get_atomic_number(atom)
#     atomic_features = np.where(atomic_features == np.tile(atom_types, (mol.GetNumAtoms(), 1)), 1, 0)
#     return atomic_features

def atomic_num_features(mol, atom_types=list(range(1, 118))):
    """_summary_

    Args:
        m (rdkit mol): Molecule to be transformed into a graph.

        atom_types (list): List of all atom types present in the dataset 
            represented by their atomic numbers.
    """
    atomic_features = np.zeros((mol.GetNumAtoms(), len(atom_types)))
    for idx, atom in enumerate(mol.GetAtoms()):
        atomic_features[idx] = get_atomic_number(atom)
    atomic_features = np.where(atomic_features == np.tile(atom_types, (mol.GetNumAtoms(), 1)), 1, 0)
    return atomic_features

def set_atommap(mol: Chem.Mol, num: int = 0) -> Chem.Mol:
    """
    Set the atom map number for all atoms in the molecule to the specified number.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.
    - num (int): The atom map number to set (default is 0).

    Returns:
    - Chem.Mol: The modified molecule object with the atom map numbers set.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def atom_equal(a1: Chem.Atom, a2: Chem.Atom) -> bool:
    """
    Check if two atoms are equal based on their symbol and formal charge.

    Parameters:
    - a1 (Chem.Atom): The first atom.
    - a2 (Chem.Atom): The second atom.

    Returns:
    - bool: True if the atoms are equal, False otherwise.
    """
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def copy_atom(atom: Chem.Atom, atommap: bool = True) -> Chem.Atom:
    """
    Create a copy of the given atom, optionally preserving its atom map number.

    Parameters:
    - atom (Chem.Atom): The atom to copy.
    - atommap (bool): If True, preserve the atom map number.

    Returns:
    - Chem.Atom: The copied atom.
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

