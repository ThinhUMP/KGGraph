from rdkit import Chem
from pathlib import Path
import sys
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGChem.chemutils import get_smiles
from KGGraph.KGGChem.features import (
    get_degree, get_total_num_hs, get_hybridization,
)

class HybridizationFeaturize:
    """
    Class to compute hybridization features for a given dataset of molecules.
    """
    #five features are in the order of (numbers of orbital s, numbers of orbital p, 
    # number of orbital d, total neighbors including Hydrogens, number of lone pairs)
    HYBRDIZATION = {
        (4,-3): [0,0,0,4,0], #AX4E0 => UNSPECIFIED => Ex: Pt smiles: N[Pt](N)(Cl)Cl
        (3,-2): [0,0,0,3,0], #AX3E0 => UNSPECIFIED => Ex: Yb smiles: Cl[Yb](Cl)Cl
        (2,-1): [0,0,0,2,0], #AX2E0 => UNSPECIFIED => Ex: Au and hybridization: UNSPECIFIED smiles: N#C[Au-]C#N
        (6,-5): [0,0,0,6,0], #AX6E0 => UNSPECIFIED => Ex: Fe and hybridization: UNSPECIFIED smiles: N#C[Fe-2](C#N)(C#N)(C#N)(C#N)N=O
        (1,-1): [1,0,0,1,0], #AX1E0 => s 
        (1,0): [1,0,0,1,0], #AX1E0 => s => Ex: Na in NaI
        (0,0): [1,0,0,0,0], #AX0E0 => s => Ex: Zn2+
        (1,1): [1,1,0,1,1], #AX1E1 => sp => Ex: N of HCN
        (2,0): [1,1,0,2,0], #AX2E0 => sp => Ex: C#C
        (0,2): [1,1,0,0,2], #AX0E2 => sp => Ex: Cr smiles: [Cr+3]
        (2,1): [1,2,0,2,1], #AX2E1 => sp2 => Ex: N of Pyrimidine
        (1,2): [1,2,0,1,2], #AX1E2 => sp2 => Ex: O of C=O
        (3,0): [1,2,0,3,0], #AX3E0 => sp2 => Ex: N of pyrrole
        (0,3): [1,2,0,0,3], #AX0E3 => sp2 => Ex: Fe2+
        (1,3): [1,2,0,1,3], #AX1E3 => sp3 => Ex: R-X (X is halogen)
        (2,2): [1,3,0,2,2], #AX2E2 => sp3 => Ex: O of R-O-R'
        (3,1): [1,2,0,3,1], #AX3E1 => sp3 => Ex: N of NR3
        (4,0): [1,3,0,4,0], #AX4E0 => sp3 => Ex: C of CR4
        (0,4): [1,2,0,0,4], #AX0E4 => sp3 => Ex: X- (X is halogen) (KI)
        (6,-2): [1,3,0,6,0], #AX6E0 => sp3 => Ex: Sb and hybridization: SP3 smiles: [SbH6+3]
        (2,3): [1,3,1,2,3], #AX2E3 => sp3d => Ex:Co
        (3,2): [1,3,1,3,2], #AX3E2 => sp3d 
        (4,1): [1,3,1,4,1], #AX4E1 => sp3d 
        (5,0): [1,3,1,5,0], #AX5E0 => sp3d => Ex: P of PCl5
        (0,5): [1,3,1,0,5], #AX0E5 => sp3d => Ex: Ag smiles: NC1=CC=C(S(=O)(=O)[N-]C2=NC=CC=N2)C=C1.[Ag+]
        (6,-1): [1,3,1,6,0], #AX6E0 => sp3d => Ex: Al smiles: NC(=O)NC1N=C(O[AlH3](O)O)NC1=O
        (4,2): [1,3,2,4,2], #AX4E2 => sp3d2
        (2,4): [1,3,2,2,4], #AX2E4 => sp3d2 => Ex: Pd in PdCl2
        (3,3): [1,3,2,3,3], #AX3E3 => sp3d2 => Ex: Dy smiles: Cl[Dy](Cl)Cl
        (5,1): [1,3,2,5,1], #AX5E1 => sp3d2
        (1,5): [1,3,2,1,5], #AX1E5 => sp3d2 => Ex:CuI
        (6,0): [1,3,2,6,0], #AX6E0 => sp3d2 => Ex: S of SF6
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
    
    def feature(atom: Chem.Atom) -> list[int, int, list[int, int, int, int, int]]:
        # print(get_smiles(atom.GetOwningMol()))
        total_single_bonds = HybridizationFeaturize.total_single_bond(atom)
        num_lone_pairs = HybridizationFeaturize.num_lone_pairs(atom)
        hybri_feat = HybridizationFeaturize.HYBRDIZATION.get((total_single_bonds, num_lone_pairs), [0,0,0,0,0])
        return total_single_bonds, num_lone_pairs, hybri_feat