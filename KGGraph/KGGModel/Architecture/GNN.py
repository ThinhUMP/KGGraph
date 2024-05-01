import torch
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn import global_add_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from .Conv.GINConv import GINConv
from .Conv.GCNConv import GCNConv

num_atom_type = 122 #including the extra motif tokens and graph token and masked atom
num_chirality_tag = 11  #degree
num_hybri_1 = 2
num_hybri_2 = 4
num_hybri_3 = 3
num_hybri_4 = 7
num_hybri_5 = 6

class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(num_hybri_1, emb_dim)
        self.x_embedding4 = torch.nn.Embedding(num_hybri_2, emb_dim)
        self.x_embedding5 = torch.nn.Embedding(num_hybri_3, emb_dim)
        self.x_embedding6 = torch.nn.Embedding(num_hybri_4, emb_dim)
        self.x_embedding7 = torch.nn.Embedding(num_hybri_5, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                pass
            elif gnn_type == "gat":
                pass
            elif gnn_type == "graphsage":
                pass

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        # + self.x_embedding2(x[:,1])
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) +\
        self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) +\
        self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) +\
        self.x_embedding7(x[:,6])

        h_list = [x]
        
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.elu(h), self.drop_ratio, training = self.training)  #relu->elu
            
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation