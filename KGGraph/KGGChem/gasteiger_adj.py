from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def add_atom_mapping(molecule):
    """
    Add atom mapping numbers to each atom in the molecule based on their index.

    Parameters:
    molecule (rdkit.Chem.Mol): The molecule to add atom mapping to.

    Returns:
    rdkit.Chem.Mol: The molecule with atom mapping numbers added.
    """
    for idx, atom in enumerate(molecule.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)  # Atom map numbers are 1-indexed in SMILES
    return molecule

def renumber_and_calculate_charges(smiles, calculate_charges=True):
    """
    Renumber atoms based on their atom map numbers and optionally calculate Gasteiger partial charges.
    Return a molecule with atom mapping and a dictionary mapping atom indices to their charges.

    Parameters:
    smiles (str): The SMILES string of the molecule.
    calculate_charges (bool): Whether to calculate Gasteiger partial charges.

    Returns:
    tuple: A molecule with atom mapping and a dictionary mapping atom indices to their charges.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = add_atom_mapping(mol)

    charges = {}
    if calculate_charges:
        AllChem.ComputeGasteigerCharges(mol)
        charges = {atom.GetIdx(): float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()}

    return mol, charges

def calculate_directed_adjacency_matrix(mol, charges):
    """
    Calculate a directed adjacency matrix for a molecule based on atomic charges.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule.
    charges (dict): A dictionary of atomic charges.

    Returns:
    numpy.ndarray: The directed adjacency matrix.
    """
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    for bond in mol.GetBonds():
        atom_0 = bond.GetBeginAtomIdx()
        atom_1 = bond.GetEndAtomIdx()
        charge_difference = charges[atom_0] - charges[atom_1]

        if charge_difference < 0:
            adjacency_matrix[atom_1, atom_0] = 1
        elif charge_difference > 0:
            adjacency_matrix[atom_0, atom_1] = 1

    return adjacency_matrix



if __name__ == '__main__':
# 1-methoxybuta-1,3-diene example (with arbitrary atom mapping)
    #smiles = "[CH2:1]=[CH:2][CH:3]=[CH:4][O:5][CH3:6]"
    smiles = "[CH2:1]=[CH:2][CH:3]=[O:4]"
    mol, charges = renumber_and_calculate_charges(smiles)
    print("Partial charges:", charges)
    directed_adj_matrix = calculate_directed_adjacency_matrix(mol, charges)
    print("Directed adjacency matrix:\n", directed_adj_matrix)