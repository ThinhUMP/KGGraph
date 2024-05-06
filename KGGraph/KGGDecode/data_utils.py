from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data
from KGGraph.KGGEncode.edge_feature import edge_feature
from KGGraph.KGGEncode.x_feature import x_feature
from KGGraph.KGGChem.atom_utils import get_mol


class MoleculeDataset(Dataset):

    def __init__(self, data_file, decompose_type, mask_node, mask_edge, fix_ratio):
        self.decompose_type = decompose_type
        self.mask_node = mask_node
        self.mask_edge = mask_edge
        self.fix_ratio = fix_ratio
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
        
        if not mask_node and not mask_edge:
            print('Not masking node and edge')
        elif not mask_node and mask_edge:
            print('Masking edge with fix ratio at 0.25', fix_ratio)
        elif mask_node and not mask_edge:
            print('Masking node with fix ratio at 0.25', fix_ratio)
        else:
            print('Masking node and edge with fix ratio at 0.25', fix_ratio)

        print('Decompose type', decompose_type)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_graph = MolGraph(smiles, self.decompose_type, self.mask_node, self.mask_edge, self.fix_ratio)
        return mol_graph

class MolGraph(object):

    def __init__(self, smiles, decompose_type, mask_node, mask_edge, fix_ratio):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.x_nosuper, self.x, self.num_part = x_feature(self.mol, decompose_type, mask_node, fix_ratio)
        self.edge_attr_nosuper, self.edge_index_nosuper, self.edge_index, self.edge_attr = edge_feature(self.mol, decompose_type, mask_edge, fix_ratio)


    def size_node(self):
        return self.x.size()[0]

    def size_edge(self):
        return self.edge_attr.size()[0]

    def size_atom(self):
        return self.x_nosuper.size()[0]

    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]

def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch
