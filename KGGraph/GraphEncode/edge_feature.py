import sys
import json
import torch
from pathlib import Path
from rdkit import Chem
from typing import Tuple
import numpy as np
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
    
# Import necessary modules and functions
from KGGraph.Chemistry.chemutils import get_smiles
from KGGraph.MotifGraph.MotitDcp.motif_decompose import MotifDecomposition, BRCISDecomposition, TreeDecomposition, SMotifDecomposition
from KGGraph.Chemistry.features import (
    is_rotatable, get_stereo, get_bond_polarity, is_bond_in_ring, 
)
from KGGraph.Chemistry.bond_type import bond_type_feature
from KGGraph.Chemistry.gasteiger_adj import renumber_and_calculate_charges, calculate_directed_adjacency_matrix

# Load bond dictionaries for process of feature extraction: bond stereo.
with open(root_dir / 'data/feature/bond_stereo_dict.json', 'r') as f:
    bond_stereo_dict = json.load(f)

class EdgeFeature:
    def __init__(self, mol: Chem.Mol, decompose_type = 'motif'):
        """
        Initializes the class with the given molecule.
        
        Parameters:
            mol: The input molecule for the class.
        """
        self.mol = mol
        self.num_bond_features = 19
        
        if decompose_type == 'motif':
            self.cliques, self.clique_edges = MotifDecomposition.defragment(mol)
        elif decompose_type == 'brics':
            self.cliques, self.clique_edges = BRCISDecomposition.defragment(mol)
        elif decompose_type == 'jin':
            self.cliques, self.clique_edges = TreeDecomposition.defragment(mol)
        elif decompose_type == 'smotif':
            self.cliques, self.clique_edges = SMotifDecomposition.defragment(mol)
        else:
            raise ValueError(f"Unknown decomposition type: {decompose_type}. It should be motif, brics, jin or smotif.")

        self.num_motif = len(self.cliques)
        self.num_atoms = mol.GetNumAtoms()
        
    
    def get_edge_node_feature(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the edge features for the molecule.

        Returns:
            edge_attr_list: A tensor of edge attributes.
            edges_index_list: A tensor of edge indices.
        """
        if len(mol.GetAtoms()) > 0:
        # Initialize lists to store edge attributes and indices
            edge_attr_list = []
            edges_index_list = []
            
            # Iterate over all bonds in the molecule
            for bond in mol.GetBonds():
                # Compute basic features for the bond
                basic_features = [
                    is_rotatable(bond),
                    get_bond_polarity(bond),
                    is_bond_in_ring(bond) #TODO: remove
                ]
                
                # Get bond type and stereo features from the dictionaries
                bond_type_features = bond_type_feature(bond)
                bond_stereo_features = bond_stereo_dict.get(get_stereo(bond), [0] * len(bond_stereo_dict))
                
                # Combine all features into a single list
                combined_features = basic_features + bond_stereo_features + bond_type_features 
                
                # Get the indices of the atoms involved in the bond
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                
                # Add the indices and features to the respective lists
                edges_index_list.extend([(i, j), (j, i)])
                edge_attr_list.extend([combined_features, combined_features])
            # Convert the lists to tensors
            edge_attr_node = torch.tensor(np.array(edge_attr_list), dtype=torch.long)
            edges_index_node = torch.tensor(np.array(edges_index_list).T, dtype=torch.long)
        else:  
            edges_index_node = torch.empty((2, 0), dtype=torch.long)
            edge_attr_node = torch.empty((0, self.num_bond_features), dtype=torch.long)
        return edge_attr_node, edges_index_node

    def get_edge_index(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct edge indices for a molecule with motif supernodes.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            torch.Tensor: Tensor representing the edge indices including motif supernodes.
        """
        _, edge_index_node = self.get_edge_node_feature(mol)

        # If there are motifs, create edges between atoms and motifs
        if self.num_motif > 0:
            # Initialize the motif_edge_index list
            motif_edge_index = []
            for k, motif_nodes in enumerate(self.cliques):
                motif_edge_index.extend([[i, self.num_atoms + k] for i in motif_nodes])
            for k in range(len(self.cliques)):
                for h in range(k+1, len(self.cliques)):
                    for bond in self.clique_edges:
                        if (bond[0] == k and bond[1] == h) or (bond[0] == h and bond[1] == k):
                            motif_edge_index.extend([[self.num_atoms + k, self.num_atoms + h], [self.num_atoms + h, self.num_atoms + k]])

            motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_node.device)

            # Create edges between motif and a supernode
            super_edge_index = [[self.num_atoms + i, self.num_atoms + self.num_motif] for i in range(self.num_motif)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_node.device)

            # Concatenate all edges
            edge_index = torch.cat((edge_index_node, motif_edge_index, super_edge_index), dim=1)
        else:
            motif_edge_index = torch.empty((0, 0), dtype=torch.long)
            # Create edges between atoms and the supernode
            super_edge_index = [[i, self.num_atoms] for i in range(self.num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_node.device)
            edge_index = torch.cat((edge_index_node, super_edge_index), dim=1)

        return motif_edge_index, super_edge_index, edge_index

    def get_edge_attr(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate edge attributes for a molecule, including attributes for edges connecting
        atoms, motifs, and a super supernode.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple containing tensors for motif edge attributes, super edge attributes,
            and the concatenated edge attributes for the entire molecular graph.
        """
        # Get the edge attributes and indices for nodes
        edge_attr_node, _ = self.get_edge_node_feature(mol)

        if self.num_motif > 0:
            # Get indices for edges connected to motifs
            motif_edge_index, _, _ = self.get_edge_index(mol)
            
            # Initialize motif edge attributes
            motif_node_edge_attr = torch.zeros((motif_edge_index.size(1)-len(self.clique_edges)*2, self.num_bond_features))
            motif_node_edge_attr[:, -3] = 1  # Set bond type for the edge between atoms and motif, 
            # we can access this feature via bond_dict json with key value 'NODEMOTIF'
            
            # Initialize motif-motif edge attributes
            motif_motif_edge_attr = torch.zeros((len(self.clique_edges)*2, self.num_bond_features))
            motif_motif_edge_attr[:, -1] = 1  # Set bond type for the edge between motif and motif
            
            # Motif edge attributes
            motif_edge_attr = torch.cat((motif_node_edge_attr, motif_motif_edge_attr), dim=0)

            # Initialize super edge attributes
            super_edge_attr = torch.zeros((self.num_motif, self.num_bond_features))
            super_edge_attr[:, -2] = 1  # Set bond type for the edge between motifs and supernode, 
            # we can access this feature via bond_dict json with key value 'MOTIFSUPERNODE'
            # Ensure that all tensors are of the same type and device
            motif_edge_attr = motif_edge_attr.to(edge_attr_node.dtype).to(edge_attr_node.device)
            super_edge_attr = super_edge_attr.to(edge_attr_node.dtype).to(edge_attr_node.device)
            # Concatenate edge attributes for the entire graph
            edge_attr = torch.cat((edge_attr_node, motif_edge_attr, super_edge_attr), dim=0)
        
        else:
            motif_edge_attr = torch.empty((0, 0))
            # Initialize super edge attributes when there are no motifs
            super_edge_attr = torch.zeros((self.num_atoms, self.num_bond_features))
            super_edge_attr[:, -4] = 1  # Set bond type for the edge between nodes and supernode, 
            # we can access this feature via bond_dict json with key value 'NODESUPERNODE'
            super_edge_attr = super_edge_attr.to(edge_attr_node.dtype).to(edge_attr_node.device)

            # # Concatenate edge attributes for the entire graph
            edge_attr = torch.cat((edge_attr_node, super_edge_attr), dim=0)
        
        return motif_edge_attr, super_edge_attr, edge_attr

def edge_feature(mol, decompose_type = 'motif'):
    obj = EdgeFeature(mol, decompose_type=decompose_type)
    _, _, edge_index = obj.get_edge_index(mol)
    _, _, edge_attr = obj.get_edge_attr(mol)
    mol_charges, charges = renumber_and_calculate_charges(get_smiles(mol))
    directed_adj_matrix = torch.tensor(calculate_directed_adjacency_matrix(mol_charges, charges), dtype=torch.long)
    return edge_index, edge_attr, directed_adj_matrix

def main():
    import time
    from joblib import Parallel, delayed
    from KGGraph import load_bace_dataset
    from tqdm import tqdm
    smiles_list, mols_list, folds, labels = load_bace_dataset('./dataset/classification/bace/raw/bace.csv')
    t1 = time.time()
    edges = Parallel(n_jobs=8)(delayed(edge_feature)(mol) for mol in tqdm(mols_list[:5]))
    t2 = time.time()
    print(t2-t1)
    # Print the results
    print(edges[0][0].size())
    print(edges[0][0])
    print(edges[0][1].size())
    print(edges[0][1])
    print(edges[0][2].size())
    print(torch.unique(edges[0][1]))

if __name__ == '__main__':
    main()
