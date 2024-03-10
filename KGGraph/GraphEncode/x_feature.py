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
from KGGraph.MotifGraph.MotitDcp.motif_decompose import MotifDecomposition
from KGGraph.Chemistry.hybridization import HybridizationFeaturize
from KGGraph.Chemistry.features import (
    get_chemical_group_block, get_atomic_number, get_period, get_group, get_atomicweight,
    get_num_valence_e, get_num_radical_electrons, get_degree, is_aromatic, is_hetero,
    is_chiral_center, get_ring_size, is_in_ring, get_ring_membership_count, 
    get_electronegativity, get_formal_charge, get_total_num_hs, get_total_valence,
    is_hydrogen_donor, is_hydrogen_acceptor, get_hybridization, get_symbol,
    is_in_aromatic_ring,
)

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
        x_node = []
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
            x_node.append(combined_features)
        
        return torch.tensor(x_node, dtype=torch.float64)
    
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

def motif_supernode_feature(mol: Chem.Mol, atom_types):
    """
    Compute motif and supernode features for a given molecule.
    
    Parameters:
        mol: The input molecule.
        atom_types: The list of atom types.
        
    Returns:
        A tuple of tensors representing motif and supernode features.
    """
    number_atom_node_attr = 42
    motif = MotifDecomposition()
    cliques = motif.defragment(mol)
    num_motif = len(cliques)

    # Pre-define tensor templates for atomic number of motif and supernode
    supernode_template = [0] * len(atom_types) + [0, 1] + [0] * (number_atom_node_attr - len(atom_types) - 2)
    motif_node_template = [0] * len(atom_types) + [1, 0] + [0] * (number_atom_node_attr - len(atom_types) - 2)

    # Create tensors based on the number of motifs
    x_supernode = torch.tensor([supernode_template], dtype=torch.long)
    if num_motif > 0:
        x_motif = torch.tensor([motif_node_template] * num_motif, dtype=torch.long)
    else:
        x_motif = torch.empty(0, number_atom_node_attr, dtype=torch.long)  # Handle cases with no motifs

    return x_motif, x_supernode

def x_feature(mol: Chem.Mol, atom_types):
    """
    Compute the feature vector for a given molecule.
    
    Parameters:
        mol: The input molecule.
        atom_types: The list of atom types.
        
    Returns:
        A tensor representing the feature vector.
    """
    atom_feature = AtomFeature(mol=mol)
    x_node = atom_feature.feature()
    x_motif, x_supernode = motif_supernode_feature(mol, atom_types)

    # Concatenate features
    x = torch.cat((x_node, x_motif.to(x_node.device), x_supernode.to(x_node.device)), dim=0)
    return x

if __name__=='__main__':
    # Load data and get atom types
    #TODO: Modify atom_types when adding MoleculeDataset class
    data = pd.read_csv('data/testcase_featurize.csv')
    smiles = data['SMILES'].tolist()
    atom_types = get_atom_types(smiles)
    mol = get_mol(smiles[0])
    x = x_feature(mol, atom_types)
    print(x)
    print(x.size())  # Print the size of the feature vector
