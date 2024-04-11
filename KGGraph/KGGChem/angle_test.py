from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

def get_atom_angle(mol, atom_idx):
    """
    Calculate the angle between the first and last neighbors of a given atom in a molecule.

    Args:
    mol (rdkit.Chem.Mol): The molecule.
    atom_idx (int): The index of the atom whose angle with its neighbors is to be calculated.

    Returns:
    float: The angle in degrees between the first and last neighbors of the atom.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    if len(neighbors) < 2:
        raise ValueError("Atom does not have enough neighbors to define an angle.")
    
    return rdMolTransforms.GetAngleDeg(mol.GetConformer(), neighbors[0], atom_idx, neighbors[-1])



def minimize_energy(mol, num_confs=10, force_field='MMFF94', random_seed=42):
    """
    Minimize the energy of a molecule using MMFF94 and ETKDGv3.

    Args:
    mol (rdkit.Chem.Mol): The molecule to be minimized.
    num_confs (int): Number of conformations to generate.
    force_field (str): The force field to use for minimization.
    etkdg_version (int): The version of ETKDG to use.

    Returns:
    rdkit.Chem.Mol: The molecule with minimized energy.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    
    for confId in range(num_confs):
        AllChem.MMFFOptimizeMolecule(mol, confId=confId, mmffVariant=force_field)
    
    return mol


if __name__=='__main__':
    water = Chem.AddHs(Chem.MolFromSmiles("O"))

    water_2 = minimize_energy(water, force_field='MMFF94')
    angle = get_atom_angle(water_2, 0)  # Calculate the angle at the oxygen atom

    print(f"The H-O-H angle in water is {angle} degrees.")
