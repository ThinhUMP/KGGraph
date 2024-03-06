from typing import Union
from rdkit import Chem
import numpy as np
import torch
from typing import List, Tuple
from rdkit.Chem.rdchem import Mol, Atom
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.chemutils import get_atom_types, get_mol
from KGGraph.Chemistry.features import *

class AtomFeature():
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
        for molecule in self.molecules:
            feature_mol = [self.compute_features(atom) for atom in molecule.GetAtoms()]
            feature_mol = torch.stack(feature_mol)
            feature_mols.append(feature_mol)
        return feature_mols
    
    def compute_features(self, atom):
        """
        Compute features for the given atom and return a tensor of features.
        """
        features = [
            get_period(atom),
            get_group(atom),
            get_atomicweight(atom),
            get_num_valence_e(atom),
            # get_chemical_group_block(atom)
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
        
    def atomic_number(self) -> List[int]:
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

class HybridizationFeaturize(AtomFeature):
    """
    Class to compute hybridization features for a given dataset of molecules.
    """
    #five features are in the order of (number of orbital s, number of orbital p, 
    # number of orbital d, total neighbors including Hydrogens, number of lone pairs)
    HYBRDIZATION = {
        (1,1): [1,1,0,1,1], #AX1E1 => sp => Ex: N of HCN
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
        Compute the total number of single bonds for a given atom.
        """
        return get_degree(atom) + get_total_num_hs(atom)

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
        return max_bond_hybridization.get(get_hybridization(atom), 0)

    @staticmethod
    def num_lone_pairs(atom: Chem.Atom) -> int:
        """
        Compute the number of lone pairs for a given atom.
        """
        return HybridizationFeaturize.num_bond_hybridization(atom) - HybridizationFeaturize.total_single_bond(atom)

    def feature(self):
        """
        Compute the hybridization feature vector for each atom in each molecule.
        """
        hybri_mols = []
        for mol in self.molecules:
            hybri_mol = []
            for atom in mol.GetAtoms():
                hybri_feat = HybridizationFeaturize.HYBRDIZATION.get((HybridizationFeaturize.total_single_bond(atom), HybridizationFeaturize.num_lone_pairs(atom)), None)
                hybri_feat = torch.tensor(hybri_feat, dtype=torch.float64)
                hybri_mol.append(hybri_feat)
            hybri_mol = torch.stack(hybri_mol)
            hybri_mols.append(hybri_mol)
        
        return hybri_mols       
        
class AtomFeaturize(AtomFeature):
    """
    Class to compute a combined feature vector for a given dataset of molecules.
    """
    def __init__(self, data, smile_col: str):
        """
        Initialize with a dataset and the column name for SMILES strings.
        """
        super().__init__(data, smile_col)
    def feature(self):
        """
        Compute the combined feature vector for each atom in each molecule.
        """
        atom_feature = []
        feature_mols = self.get_feature()
        atomic_number_mols = self.atomic_number()
        hybri_mols = HybridizationFeaturize(self.data, self.smiles_col).feature()
        for i in range(len(self.smiles)):
            atom_feature.append(torch.cat((feature_mols[i], atomic_number_mols[i], hybri_mols[i]), dim=1))
        return atom_feature

if __name__=='__main__':
    import pandas as pd
    import sys
    import pathlib
    root_dir = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(root_dir)
    data = pd.read_csv('data/testcase_featurize.csv')
    atom_feature_obj = AtomFeaturize(data=data, smile_col='SMILES')
    atom_features = atom_feature_obj.feature()
    print(atom_features[0].shape)
    print(len(atom_features))
