import torch
import numpy as np
from rdkit import Chem
from pathlib import Path
import sys
from typing import List
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))

from KGGraph.Chemistry.chemutils import get_atom_types, get_smiles
from KGGraph.MotifGraph.MotitDcp import MotifDecomposition, BRCISDecomposition, TreeDecomposition, SMotifDecomposition
from KGGraph.Chemistry.hybridization import HybridizationFeaturize
from KGGraph.Chemistry.features import (
    get_chemical_group_block, get_period, get_group, get_atomicweight,
    get_num_valence_e, get_num_radical_electrons, get_degree, is_aromatic, is_hetero,
    is_chiral_center, get_ring_size, is_in_ring, get_ring_membership_count, 
    get_electronegativity, get_formal_charge, get_total_num_hs, get_total_valence,
    is_hydrogen_donor, is_hydrogen_acceptor, get_hybridization, get_symbol,
    is_in_aromatic_ring,
)
from KGGraph.Chemistry.chemutils import get_atom_types, atomic_num_features

class AtomFeature:
    """
    Class to compute atom features for a given dataset of molecules.
    """
    def __init__(self, mol: Chem.Mol, atom_types: List[int]):
        """
        Initializes the class with the given molecule.
        
        Parameters:
            mol: The input molecule for the class.
        """
        self.mol = mol
        self.atom_types = atom_types
        
    def feature(self):
        """
        Get feature molecules from the list of molecules and return a list of feature molecules.
        """
        x_node_list = []
        atomic_features = atomic_num_features(self.mol, self.atom_types)
        
        # Atom feature dictionary for each atom in the molecule with key as atom index and value as atom features
        atom_feature_dic = {}
        
        for atom in self.mol.GetAtoms():
            # basic_features = self.compute_basic_features(atom)
            chemical_group = get_chemical_group_block(atom)

            total_single_bonds, num_lone_pairs, hybri_feat = HybridizationFeaturize.feature(atom)
            if hybri_feat == [0,0,0,0,0]:
                print(f'Error key:{(total_single_bonds, num_lone_pairs)} with atom: {get_symbol(atom)} and hybridization: {get_hybridization(atom)} smiles: {get_smiles(self.mol)}')
            
            combined_features = [get_degree(atom)] + chemical_group + hybri_feat
            
            # Add atom feature to dictionary to use for motif feature extraction
            atom_feature = np.concatenate((combined_features, atomic_features[atom.GetIdx()]), axis=0)
            atom_feature_tensor = torch.tensor(atom_feature, dtype=torch.long)
            atom_feature_dic[atom.GetIdx()] = atom_feature_tensor
            
            x_node_list.append(combined_features)
        
        x_node_array = np.array(x_node_list)
        x_node = torch.tensor(np.concatenate((x_node_array, atomic_features), axis=1), dtype=torch.long)
        
        return x_node, atom_feature_dic
    
    def compute_basic_features(self, atom) -> List:
        """
        Compute basic features for the given atom and return a tensor of features.
        """
        basic_features = [
            # get_period(atom),
            # get_group(atom),
            # get_atomicweight(atom),
            # get_num_valence_e(atom),
            # is_chiral_center(atom),
            # get_formal_charge(atom),
            # get_total_num_hs(atom),
            # get_total_valence(atom),
            # get_num_radical_electrons(atom),
            # get_degree(atom),
            # int(is_aromatic(atom)),
            # int(is_hetero(atom)),
            # int(is_hydrogen_donor(atom)),
            # int(is_hydrogen_acceptor(atom)),
            # get_ring_size(atom),
            # int(is_in_ring(atom)),
            # get_ring_membership_count(atom),
            # int(is_in_aromatic_ring(atom)),
            # get_electronegativity(atom),
        ]
        return basic_features

def motif_supernode_feature(mol: Chem.Mol, number_atom_node_attr: int, atom_feature_dic: dict, decompose_type):
    """
    Compute motif and supernode features for a given molecule.
    
    Parameters:
        mol: The input molecule.
        number_atom_node_attr: number of atom features in the molecule.
        atom_feature_dic: The dictionary of atom features in the molecule.
        
    Returns:
        A tuple of tensors representing motif and supernode features.
    """
    if decompose_type == 'motif':
        cliques, _ = MotifDecomposition.defragment(mol)
    elif decompose_type == 'brics':
        cliques, _ = BRCISDecomposition.defragment(mol)
    elif decompose_type == 'jin':
        cliques, _ = TreeDecomposition.defragment(mol)
    elif decompose_type == 'smotif':
        cliques, _ = SMotifDecomposition.defragment(mol)
    else:
        raise ValueError(f"Unknown decomposition type: {decompose_type}. It should be motif, brics, jin or smotif.")
    
    num_motif = len(cliques)
    x_motif = []
        
    
    if num_motif > 0:
        for k, motif_nodes in enumerate(cliques):
            motif_node_feature = torch.zeros(number_atom_node_attr, dtype=torch.long)
            for i in motif_nodes:
                motif_node_feature += atom_feature_dic[i]
            x_motif.append(motif_node_feature)

        x_motif = torch.stack(x_motif, dim = 0)
        x_supernode = torch.sum(x_motif, dim=0).unsqueeze(0)
    else:
        x_motif = torch.empty(0, number_atom_node_attr, dtype=torch.long) # Handle cases with no motifs
        x_supernode = torch.zeros(number_atom_node_attr, dtype=torch.long)
        for i in atom_feature_dic.keys():
            x_supernode += atom_feature_dic[i]
        x_supernode = x_supernode.unsqueeze(0) 
        
    return x_motif, x_supernode


def x_feature(mol: Chem.Mol, atom_types: List[int], decompose_type):
    """
    Compute the feature vector for a given molecule.
    
    Parameters:
        mol: The input molecule.
        atom_types: The list of atom types.
        
    Returns:
        A tensor representing the feature vector.
    """
    atom_feature = AtomFeature(mol=mol, atom_types=atom_types)
    x_node, atom_feature_dic = atom_feature.feature()
    x_motif, x_supernode = motif_supernode_feature(mol, number_atom_node_attr=x_node.size(1), atom_feature_dic=atom_feature_dic, decompose_type = decompose_type)

    # Concatenate features
    x = torch.cat((x_node, x_motif.to(x_node.device), x_supernode.to(x_node.device)), dim=0)

    return x

def main():
    from joblib import Parallel, delayed
    import time
    from tqdm import tqdm
    from KGGraph import load_clintox_dataset
    from pathlib import Path
    import sys
    import pandas as pd
    from typing import List
    # Get the root directory
    root_dir = Path(__file__).resolve().parents[2]
    # Add the root directory to the system path
    sys.path.append(str(root_dir))
    from KGGraph import load_bace_dataset
    # data = pd.read_csv('./dataset/classification/clintox/raw/clintox.csv')
    # smiles = data['smiles'].tolist()[:10]
    # mols = [get_mol(smile) for smile in smiles]
    smiles, mols, folds, labels = load_bace_dataset('dataset/classification/bace/raw/bace.csv')
    atom_types = get_atom_types(smiles)
    t1 = time.time()
    x = Parallel(n_jobs=-1)(delayed(x_feature)(mol, atom_types) for mol in tqdm(mols))
    t2 = time.time()
    print(t2-t1)
    print(x[0])
    print(x[0].size())  # Print the size of the feature vector

if __name__=='__main__':
    main()
