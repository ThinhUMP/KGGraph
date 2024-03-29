from rdkit import Chem
import json
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.features import get_bond_type, is_conjugated
    
with open(root_dir+'/data/feature/bond_dict.json', 'r') as f:
    bond_dict = json.load(f)

def bond_type_feature(bond):
    """
    Determine the feature of a bond based on its type and the atoms it connects.

    Parameters:
    bond (rdkit.Chem.rdchem.Bond): The bond to analyze.

    Returns:
    list: A list representing the bond type feature.
    """
    if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
        return bond_dict.get(get_bond_type(bond))
    elif is_conjugated(bond):
        return bond_dict.get('CONJUGATE')
    else:
        return bond_dict.get(get_bond_type(bond),  [0, 0, 0, 0, 1, 0, 0, 0, 0])  # Return other type if not found