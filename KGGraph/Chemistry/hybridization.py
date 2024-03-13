from rdkit import Chem
from pathlib import Path
import sys
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))

from KGGraph.Chemistry.features import (
    get_degree, get_total_num_hs, get_hybridization,
)


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
    
    def feature(atom: Chem.Atom) -> list:
        total_single_bonds = HybridizationFeaturize.total_single_bond(atom)
        num_lone_pairs = HybridizationFeaturize.num_lone_pairs(atom)
        hybri_feat = HybridizationFeaturize.HYBRDIZATION.get((total_single_bonds, num_lone_pairs), None)
        return total_single_bonds, num_lone_pairs, hybri_feat