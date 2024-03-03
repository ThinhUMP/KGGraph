from typing import Union
from rdkit import Chem
import numpy as np
import torch
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.chemutils import *
from KGGraph.Chemistry.features import (
    is_chiral_center, get_formal_charge, get_total_num_hs, get_total_valence, 
    get_num_radical_electrons, get_degree, is_aromatic, is_hetero, is_hydrogen_donor, 
    is_hydrogen_acceptor, get_ring_size, is_in_ring, get_ring_membership_count, 
    is_in_aromatic_ring, get_electronegativity,
)

class AtomFeature():
    
    def __init__(self, data, smile_col: str):
        self.data = data
        self.smiles_col = smile_col
        self.smiles = data[smile_col].tolist()
        self.molecules = [get_mol(smile)for smile in self.smiles]
        
    def get_feature(self):
        feature_mols = []
        for molecule in self.molecules:
            feature_mol = []
            for atom in molecule.GetAtoms():
                chiral_center = is_chiral_center(atom)
                formal_charge = get_formal_charge(atom)
                total_num_hs = get_total_num_hs(atom)
                total_valence = get_total_valence(atom)
                num_radical_electrons = get_num_radical_electrons(atom)
                degree = get_degree(atom)
                aromatic = is_aromatic(atom)
                hetero = is_hetero(atom)
                hydrogen_donor = is_hydrogen_donor(atom)
                hydrogen_acceptor = is_hydrogen_acceptor(atom)
                ring_size = get_ring_size(atom)
                in_ring = is_in_ring(atom)
                ring_membership_count = get_ring_membership_count(atom)
                in_aromatic_ring = is_in_aromatic_ring(atom)
                electronegativity = get_electronegativity(atom)

                # Print out each variable and its value
                variables = [chiral_center, formal_charge, total_num_hs, total_valence,
                            num_radical_electrons, degree, aromatic, hetero, hydrogen_donor, 
                            hydrogen_acceptor, ring_size, in_ring, ring_membership_count, 
                            in_aromatic_ring, electronegativity]
                feature_atom = torch.tensor(variables, dtype=torch.float64)
                
                feature_mol.append(feature_atom)
            
            feature_mol = torch.stack(feature_mol)
            feature_mols.append(feature_mol)
        
        return feature_mols
        
    def atomic_number(self) -> List[int]:
        atom_types = get_atom_types(self.smiles)
        atomic_number_mols = []
        for mol in self.molecules:
            atomic_number = np.zeros((mol.GetNumAtoms(), len(atom_types)))
            for index, atom in enumerate(mol.GetAtoms()):
                atomic_number[index] = atom.GetAtomicNum()
            atomic_number = np.where(atomic_number == np.tile(atom_types, (mol.GetNumAtoms(), 1)), 1, 0)
            atomic_number = torch.tensor(atomic_number, dtype=torch.float64)        
            atomic_number_mols.append(atomic_number)
        return atomic_number_mols
        
        
if __name__=='__main__':
    import pandas as pd
    import sys
    import pathlib
    root_dir = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(root_dir)
    data = pd.read_csv('data/check.csv')
    check_feature = AtomFeature(data=data, smile_col = 'SMILES')
    features = check_feature.get_feature()
    atomic_num = check_feature.atomic_number()
    print(features[0].shape)
    print(len(features))
    print(atomic_num[0].shape)
    print(len(atomic_num))
    # print(feature)
    # print(check.smile)
    # print(check.molecule)