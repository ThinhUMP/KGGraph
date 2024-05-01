import torch
import math
import random
from copy import deepcopy
import numpy as np
from rdkit import Chem
from pathlib import Path
import sys
from typing import List
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))

from KGGraph.KGGChem.atom_utils import get_smiles
from KGGraph.KGGDecompose.MotitDcp.brics_decompose import BRCISDecomposition
from KGGraph.KGGDecompose.MotitDcp.jin_decompose import TreeDecomposition
from KGGraph.KGGDecompose.MotitDcp.motif_decompose import MotifDecomposition
from KGGraph.KGGDecompose.MotitDcp.smotif_decompose import SMotifDecomposition
from KGGraph.KGGChem.hybridization import HybridizationFeaturize
from KGGraph.KGGChem.atom_features import (
    get_degree, get_hybridization, get_symbol, get_atomic_number,
)
from KGGraph.KGGChem.atom_utils import atomic_num_features

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(0, 119)),
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

class AtomFeature:
    """
    Class to compute atom features for a given dataset of molecules.
    """

    @staticmethod
    def feature(mol: Chem.Mol):
        """
        Get feature molecules from the list of molecules and return a list of feature molecules.
        """
        x_node_list = []
        
        for atom in mol.GetAtoms():
            total_sigma_bonds, num_lone_pairs, hybri_feat = HybridizationFeaturize.feature(atom)
            if hybri_feat == [0,0,0,0,0]:
                print(f'Error key:{(total_sigma_bonds, num_lone_pairs)} with atom: {get_symbol(atom)} and hybridization: {get_hybridization(atom)} smiles: {get_smiles(mol)}')
            
            atom_feature = [allowable_features['possible_atomic_num_list'].index(
                get_atomic_number(atom))] + [allowable_features['possible_degree_list'].index(get_degree(atom))] + hybri_feat
            
            x_node_list.append(atom_feature)
        
        x_node_array = np.array(x_node_list)
        x_node = torch.tensor(x_node_array, dtype=torch.long)
        
        return x_node
    
    @staticmethod
    def masked_atom_feature(mol: Chem.Mol, x_node, fix_ratio):
        
        num_node = mol.GetNumAtoms()
        if not fix_ratio:
            num_masked_node = max([1, math.floor(0.25*num_node)])
        else:
            num_masked_node = math.floor(0.25*num_node)

        masked_node = random.sample(list(range(num_node)), num_masked_node)

        x_node_masked = deepcopy(x_node)
        for atom_idx in masked_node:
            x_node_masked[atom_idx, :] = torch.tensor([121,0,0,0,0,0,0]) #121 implies masked atom

        return x_node_masked

def motif_supernode_feature(mol: Chem.Mol, number_atom_node_attr: int, decompose_type):
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
        template_motif = torch.zeros((1, number_atom_node_attr), dtype=torch.long)
        template_motif[0, 0] = 120
        x_motif = template_motif.repeat_interleave(num_motif, dim=0)
        x_supernode = torch.zeros((1, number_atom_node_attr), dtype=torch.long)
        x_supernode[0, 0] = 119
    else:
        x_motif = torch.empty(0, number_atom_node_attr, dtype=torch.long) # Handle cases with no motifs
        x_supernode = torch.zeros((1, number_atom_node_attr), dtype=torch.long)
        x_supernode[0, 0] = 119
        
    return x_motif, x_supernode


def x_feature(mol: Chem.Mol, decompose_type, pretrain, fix_ratio):
    """
    Compute the feature vector for a given molecule.
    
    Parameters:
        mol: The input molecule.
        atom_types: The list of atom types.
        
    Returns:
        A tensor representing the feature vector.
    """
    atom_feature = AtomFeature()
    x_node = AtomFeature.feature(mol=mol)
    x_motif, x_supernode = motif_supernode_feature(mol, number_atom_node_attr=x_node.size(1), decompose_type = decompose_type)

    if not pretrain:
        # Concatenate features
        x = torch.cat((x_node, x_motif.to(x_node.device), x_supernode.to(x_node.device)), dim=0).to(torch.long)
    else:
        x_node_masked = AtomFeature.masked_atom_feature(mol=mol, x_node=x_node, fix_ratio=fix_ratio)
        x = torch.cat((x_node_masked, x_motif.to(x_node.device), x_supernode.to(x_node.device)), dim=0).to(torch.long)
    
    num_part = (x_node.size(0), x_motif.size(0), x_supernode.size(0))
    
    return x_node, x, num_part

def main():
    from joblib import Parallel, delayed
    import time
    from tqdm import tqdm
    from pathlib import Path
    import sys
    # Get the root directory
    root_dir = Path(__file__).resolve().parents[2]
    # Add the root directory to the system path
    sys.path.append(str(root_dir))
    from KGGraph.KGGProcessor.loader import load_bace_dataset
    _, mols,folds, _ = load_bace_dataset('Data/classification/bace/raw/bace.csv')
    t1 = time.time()
    # results = Parallel(n_jobs=-1)(delayed(x_feature)(mol, decompose_type='motif') for mol in tqdm(mols))
    for mol in mols:
        x_node, x, num_part = x_feature(mol, decompose_type='motif', pretrain=True)
        print(x)
    t2 = time.time()
    print(t2-t1)
    # print(results[0][0])
    # print(results[0][1])
    # print(results[0][2])

if __name__=='__main__':
    main()
