import torch
import numpy as np
from tqdm import tqdm
import time
from typing import List
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.chemutils import get_atom_types, get_mol
from KGGraph.Chemistry.features import *

class GetFeature():
    """
    Class to compute atom features for a given dataset of molecules.
    """
    def __init__(self, data, smile_col: str):
        """
        Initializes the class with the given data and smile column.
        
        Parameters:
            data: The input data for the class.
            smile_col: The name of the column containing smiles.
            
        Returns:
            None
        """
        self.data = data
        self.smiles_col = smile_col
        self.smiles = data[smile_col].tolist()
        self.molecules = [get_mol(smile) for smile in self.smiles]
        
    def get_feature(self):
        """
        Get feature molecules from the list of molecules and return a list of feature molecules.
        """
        feature_mols = []
        for molecule in tqdm(self.molecules):
            feature_mol = torch.stack([self.compute_features(atom) for atom in molecule.GetAtoms()])
            feature_mols.append(feature_mol)
        return feature_mols
    
    def compute_features(self, atom) -> torch.Tensor:
        """
        Compute features for the given atom and return a tensor of features.
        """
        features = [
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
            is_aromatic(atom),
            is_hetero(atom),
            is_hydrogen_donor(atom),
            is_hydrogen_acceptor(atom),
            get_ring_size(atom),
            is_in_ring(atom),
            get_ring_membership_count(atom),
            is_in_aromatic_ring(atom),
            get_electronegativity(atom),
        ]
        return torch.tensor(features, dtype=torch.float64)
    
    def chemical_group(self) -> List[torch.Tensor]:
        """
        This function iterates through the molecules in the class instance and retrieves the chemical group block for each atom in each molecule. It returns a list of chemical group blocks for all atoms in all molecules as tensors.
        """
        group_block_mols = []
        for mol in self.molecules:
            group_blocks = [get_chemical_group_block(atom) for atom in mol.GetAtoms()]
            group_blocks = torch.stack(group_blocks)
            group_block_mols.append(group_blocks)
        return group_block_mols

    def atomic_number(self) -> List[List[float]]:
        """
        Compute a one-hot encoding of the atomic number for each atom in each molecule.
        """
        atom_types = get_atom_types(self.smiles) #get all types of atom in the dataset
        atomic_number_mols = []
        for mol in self.molecules:
            atomic_number = np.zeros((mol.GetNumAtoms(), len(atom_types)))
            for index, atom in enumerate(mol.GetAtoms()):
                atomic_number[index] = atom.GetAtomicNum()
            atomic_number = np.where(atomic_number == np.tile(atom_types, (mol.GetNumAtoms(), 1)), 1, 0)
            atomic_number = torch.tensor(atomic_number, dtype=torch.float64)        
            atomic_number_mols.append(atomic_number)
        return atomic_number_mols

class HybridizationFeaturize(GetFeature):
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

    def __init__(self, data, smile_col: str):
        """
        Initialize with a dataset and the column name for SMILES strings.
        """
        super().__init__(data, smile_col)
    @staticmethod
    def total_single_bond(atom: Chem.Atom) -> int:
        """
        Compute the total number of single bonds for a given atom including the bond with hydrogens.
        """
        total_single_bonds = get_degree(atom) + get_total_num_hs(atom)
        return total_single_bonds

    @staticmethod
    def num_bond_hybridization(atom: Chem.Atom) -> int: #including hydrogens
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

    def feature(self):
        """
        Compute the hybridization feature vector for each atom in each molecule.
        """
        hybri_mols = []
        for mol in self.molecules:
            hybri_mol = []
            for atom in mol.GetAtoms():
                total_single_bonds = HybridizationFeaturize.total_single_bond(atom)
                num_lone_pairs = HybridizationFeaturize.num_lone_pairs(atom)
                hybri_feat = HybridizationFeaturize.HYBRDIZATION.get((total_single_bonds, num_lone_pairs), None)
                if hybri_feat is None:
                    raise ValueError(f'Error key:{(total_single_bonds, num_lone_pairs)} with atom: {get_symbol(atom)} and hybridization: {get_hybridization(atom)}')
                hybri_feat = torch.tensor(hybri_feat, dtype=torch.float64)
                hybri_mol.append(hybri_feat)
            hybri_mol = torch.stack(hybri_mol)
            hybri_mols.append(hybri_mol)
        
        return hybri_mols       
        
class AtomFeature(GetFeature):
    """
    Class to compute a combined feature vector for a given dataset of molecules.
    """
    def __init__(self, data, smile_col: str):
        """
        Initialize with a dataset and the column name for SMILES strings.
        """
        super().__init__(data, smile_col)

    def feature(self) -> List[torch.Tensor]:
        """
        Compute the combined feature vector for each atom in each molecule.
        """
        start_time = time.time()

        atom_feature = []
        feature_mols = self.get_feature()
        atomic_number_mols = self.atomic_number()
        group_block_mols = self.chemical_group()
        hybri_mols = HybridizationFeaturize(self.data, self.smiles_col).feature()
        
        for i in range(len(self.smiles)):
            combined_features = torch.cat((feature_mols[i], atomic_number_mols[i], hybri_mols[i], group_block_mols[i]), dim=1)
            atom_feature.append(combined_features)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return atom_feature

if __name__=='__main__':
    import pandas as pd
    import sys
    import pathlib
    root_dir = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(root_dir)
    data = pd.read_csv('data/testcase_featurize.csv')
    atom_feature_obj = AtomFeature(data=data, smile_col='SMILES')
    atom_features = atom_feature_obj.feature()
    print(atom_features[0].shape)
    print(len(atom_features))
