import torch
import numpy as np
from rdkit import Chem
from pathlib import Path
import sys
import pandas as pd
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))

from KGGraph.Chemistry.chemutils import get_atom_types, get_mol
from KGGraph.Chemistry.features import (
    get_chemical_group_block, get_atomic_number, get_period, get_group, get_atomicweight,
    get_num_valence_e, get_num_radical_electrons, get_degree, is_aromatic, is_hetero,
    is_chiral_center, get_ring_size, is_in_ring, get_ring_membership_count, 
    get_electronegativity, get_formal_charge, get_total_num_hs, get_total_valence,
    is_hydrogen_donor, is_hydrogen_acceptor, get_hybridization, get_symbol,
    is_in_aromatic_ring,
)

# Load data and get atom types
#TODO: Modify atom_types when adding MoleculeDataset class
data = pd.read_csv('data/testcase_featurize.csv')
smiles = data['SMILES'].tolist()
atom_types = get_atom_types(smiles)

class AtomFeature:
    """
    Class to compute atom features for a given dataset of molecules.
    """
    def __init__(self, mol: Chem.Mol):
        """
        Initializes the class with the given molecule.
        
        Parameters:
            mol: The input molecule for the class.
        """
        self.mol = mol
        
    def feature(self):
        """
        Get feature molecules from the list of molecules and return a list of feature molecules.
        """
        atomic_number = np.zeros((self.mol.GetNumAtoms(), len(atom_types)))
        x_node_atom = []
        for index, atom in enumerate(self.mol.GetAtoms()):
            basic_features = self.compute_basic_features(atom)
            chemical_group = get_chemical_group_block(atom)
            
            atomic_number[index] = get_atomic_number(atom)
            atomic_number = np.where(atomic_number == np.tile(atom_types, (self.mol.GetNumAtoms(), 1)), 1, 0)
            # Add two zeros to represent the motif and supernode atomic number
            atomic_number_super = np.concatenate((atomic_number, np.zeros((self.mol.GetNumAtoms(), 2))), axis=1)
            
            total_single_bonds = HybridizationFeaturize.total_single_bond(atom)
            num_lone_pairs = HybridizationFeaturize.num_lone_pairs(atom)
            hybri_feat = HybridizationFeaturize.HYBRDIZATION.get((total_single_bonds, num_lone_pairs), None)
            if hybri_feat is None:
                raise ValueError(f'Error key:{(total_single_bonds, num_lone_pairs)} with atom: {get_symbol(atom)} and hybridization: {get_hybridization(atom)}')
            
            combined_features = basic_features + chemical_group + hybri_feat + atomic_number_super[index].tolist()
            x_node_atom.append(combined_features)
        
        return torch.tensor(x_node_atom, dtype=torch.float64)
    
    def compute_basic_features(self, atom) -> torch.Tensor:
        """
        Compute basic features for the given atom and return a tensor of features.
        """
        basic_features = [
            get_period(atom),
            get_group(atom),
            get_atomicweight(atom),
            get_num_valence_e(atom),
            is_chiral_center(atom),
            get_formal_charge(atom),
            get_total_num_hs(atom),
            get_total_valence(atom),
            get_num_radical_electrons(atom),
            get_degree(atom),
            int(is_aromatic(atom)),
            int(is_hetero(atom)),
            int(is_hydrogen_donor(atom)),
            int(is_hydrogen_acceptor(atom)),
            get_ring_size(atom),
            int(is_in_ring(atom)),
            get_ring_membership_count(atom),
            int(is_in_aromatic_ring(atom)),
            get_electronegativity(atom),
        ]
        return basic_features

class HybridizationFeaturize:
    """
    Class to compute hybridization features for a given dataset of molecules.
    """
    #five features are in the order of (numbers of orbital s, numbers of orbital p, 
    # number of orbital d, total neighbors including Hydrogens, number of lone pairs)
    HYBRDIZATION = {
        (1,1): [1,1,0,1,1], #AX1E1 => sp => Ex: N of HCN
        (2,0): [1,1,0,2,0], #AX2E0 => sp => Ex: C#C
        (2,1): [1,2,0,2,1], #AX2E1 => sp2 => Ex: N of Pyrimidine
        (1,2): [1,2,0,1,2], #AX1E2 => sp2 => Ex: O of C=O
        (3,0): [1,2,0,3,0], #AX1E1 => sp2 => Ex: N of pyrrole
        (1,3): [1,2,0,1,3], #AX1E3 => sp3 => Ex: R-X (X is halogen)
        (2,2): [1,3,0,2,2], #AX2E2 => sp3 => Ex: O of R-O-R'
        (3,1): [1,2,0,3,1], #AX3E1 => sp3 => Ex: N of NR3
        (4,0): [1,3,0,4,0], #AX1E0 => sp3 => Ex: C of CR4
        (3,2): [1,3,1,3,2], #AX1E2 => sp3d 
        (4,1): [1,3,1,4,1], #AX1E1 => sp3d 
        (5,0): [1,3,1,5,0], #AX1E0 => sp3d => Ex: P of PCl5
        (4,2): [1,3,2,4,2], #AX1E2 => sp3d2 
        (5,1): [1,3,2,5,1], #AX1E1 => sp3d2 
        (6,0): [1,3,2,6,0], #AX1E0 => sp3d2 => Ex: S of SF6
    }

    @staticmethod
    def total_single_bond(atom: Chem.Atom) -> int:
        """
        Compute the total number of single bonds for a given atom including the bond with hydrogens.
        """
        total_single_bonds = get_degree(atom) + get_total_num_hs(atom)
        return total_single_bonds

    @staticmethod
    def num_bond_hybridization(atom: Chem.Atom) -> int:
        """
        Compute the number of bonds involved in hybridization for a given atom.
        """
        max_bond_hybridization = {
            'SP3D2': 6,
            'SP3D': 5,
            'SP3': 4,
            'SP2': 3,
            'SP': 2,
            'UNSPECIFIED': 1,
        }
        
        num_bonds_hybridization = max_bond_hybridization.get(get_hybridization(atom), 0)
        return num_bonds_hybridization

    @staticmethod
    def num_lone_pairs(atom: Chem.Atom) -> int:
        """
        Compute the number of lone pairs for a given atom.
        """
        num_lone_pairs = HybridizationFeaturize.num_bond_hybridization(atom) - HybridizationFeaturize.total_single_bond(atom)
        return num_lone_pairs     

if __name__=='__main__':
    data = pd.read_csv(root_dir / 'data/testcase_featurize.csv')
    smile = data['SMILES'][0]
    mol = get_mol(smile)
    atom_feature_obj = AtomFeature(mol = mol)
    atom_features = atom_feature_obj.feature()
    print(atom_features.shape)
    print(atom_features)
