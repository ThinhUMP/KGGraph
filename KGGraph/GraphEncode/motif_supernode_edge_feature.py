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
from KGGraph.GraphEncode.x_feature import AtomFeature

# Load data and get atom types
#TODO: Modify atom_types when adding MoleculeDataset class
data = pd.read_csv('./data/testcase_featurize.csv')
smiles = data['SMILES'].tolist()
atom_types = get_atom_types(smiles)




# cliques = motif_decomp(mol)
#     num_motif = len(cliques)
#     if num_motif > 0:
#         motif_x = torch.tensor([[120, 0]]).repeat_interleave(num_motif, dim=0).to(x_nosuper.device)
#         x = torch.cat((x_nosuper, motif_x, super_x), dim=0)

#         motif_edge_index = []
#         for k, motif in enumerate(cliques):
#             motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
#         motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)

#         super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
#         super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
#         edge_index = torch.cat((edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

#         motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 2)
#         motif_edge_attr[:,0] = 6 
#         motif_edge_attr = motif_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)

#         super_edge_attr = torch.zeros(num_motif, 2)
#         super_edge_attr[:,0] = 5 
#         super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
#         edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

#         num_part = (num_atoms, num_motif, 1)

#     else:
#         x = torch.cat((x_nosuper, super_x), dim=0)

#         super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
#         super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
#         edge_index = torch.cat((edge_index_nosuper, super_edge_index), dim=1)

#         super_edge_attr = torch.zeros(num_atoms, 2)
#         super_edge_attr[:,0] = 5 #bond type for self-loop edge
#         super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
#         edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim = 0)

#         num_part = (num_atoms, 0, 1)