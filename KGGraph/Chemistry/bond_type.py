from rdkit import Chem
import json
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.features import get_bond_type, is_conjugated
    
with open(root_dir+'/dataset/feature/bond_dict.json', 'r') as f:
    bond_dict = json.load(f)

def bond_type_feature(bond):
    """
    Determine the feature representation of a bond in a molecular structure. The function 
    categorizes the bond into specific types, such as aromatic or conjugated, and returns 
    a corresponding feature vector from a predefined dictionary.

    Parameters:
    bond (Chem.rdchem.Bond): The bond to analyze, an instance of RDKit's Bond class.

    Returns:
    list: A feature list representing the bond type. The feature is a predefined vector
    from a dictionary based on the bond type.
    """
    
    # Get the bond type as a string representation
    bond_type = get_bond_type(bond)

    # Check for aromatic bond type
    if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
        return bond_dict.get(bond_type)
    
    # Check for conjugated bond type
    if is_conjugated(bond):
        return bond_dict.get('CONJUGATE')
    
    # Return the bond type feature or a default 'other' type feature vector
    return bond_dict.get(bond_type, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

# Auxiliary functions and dictionary must be defined:
# get_bond_type(bond): Should return a string representation of the bond type.
# is_conjugated(bond): Should return a boolean indicating if the bond is conjugated.
# bond_dict: A dictionary mapping bond type strings to feature lists.
