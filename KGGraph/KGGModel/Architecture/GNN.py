import torch
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn import global_add_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from .Conv.GINConv import GINConv
from .Conv.GCNConv import GCNConv
from .vocab_x_embedding import num_vocab_x_embedding

vocab_x_embedding = 119
class GNN(torch.nn.Module):
    """
    A generalized graph neural network (GNN) module that supports various GNN types and jump knowledge (JK) concatenation methods.

    This module is designed to generate node representations by processing node features and graph structure through 
    multiple GNN layers. It supports various GNN architectures and offers different methods for aggregating node 
    representations across layers.

    Args:
        dataset: The dataset object, which is used to derive the number of node features or embeddings.
        num_layer (int): The number of GNN layers.
        emb_dim (int): The dimensionality of the node embeddings.
        JK (str): The method for aggregating node representations across layers. Options include 'last', 'concat', 'max', or 'sum'.
        drop_ratio (float): The dropout rate applied to the node embeddings.
        gnn_type (str): The type of GNN layer to use. Options include 'gin', 'gcn', 'graphsage', and 'gat'.
    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        """
        Initializes the GNN module with the specified architecture and parameters.

        Args:
            dataset: The dataset object, which is used to derive the number of node features or embeddings.
            num_layer (int): The number of GNN layers.
            emb_dim (int): The dimensionality of the node embeddings.
            JK (str): The aggregation method for node representations across layers.
            drop_ratio (float): The dropout rate applied to the node embeddings.
            gnn_type (str): The type of GNN layer to use.
        """
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        # vocab_x_embedding = num_vocab_x_embedding(dataset)
        self.x_embedding = torch.nn.Embedding(vocab_x_embedding, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            else:
                raise ValueError("Undefined GNN type.")

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        """
        Forward pass of the GNN module. It can take a data object or the individual components as input.

        This method processes the input through multiple GNN layers, applies batch normalization and dropout, 
        and aggregates the node representations according to the specified JK method.

        Args:
            *argv: Variable length argument list. Can be a single data object or three separate components 
                   of the data object (x, edge_index, edge_attr).

        Returns:
            Tensor: The node representations generated by the GNN.
        """
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # x_clamped = torch.clamp(x, 0, self.x_embedding.num_embeddings - 1)
        x_embeddings = self.x_embedding(x).sum(dim=1)
        
        h_list = [x_embeddings]
        
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