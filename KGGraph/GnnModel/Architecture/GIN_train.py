import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
from .GNN import GNN

class GINTrain(torch.nn.Module):
    """
    A GIN model extension that incorporates edge information by concatenation for graph-level prediction tasks.

    This class defines a GNN model which can handle node features, edge features, and graph connectivity to
    produce embeddings for graph-level prediction tasks. It supports different types of GNN layers (e.g., GIN, GCN)
    and aggregation methods for node representations.

    Args:
        num_layer (int): The number of GNN layers.
        emb_dim (int): The dimensionality of node embeddings.
        num_tasks (int): The number of tasks for multi-task learning, typically corresponding to the number of output features.
        JK (str): Choice of how to aggregate node representations across layers. Options are 'last', 'concat', 'max', or 'sum'.
        drop_ratio (float): The dropout rate applied after GNN layers.
        gnn_type (str): The type of GNN layer to use. Options include 'gin', 'gcn', 'graphsage', and 'gat'.
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        """
        Initializes the GINTrain model with the specified architecture and parameters.

        Args:
            num_layer (int): The number of GNN layers.
            emb_dim (int): The dimensionality of node embeddings.
            num_tasks (int): The number of tasks for multi-task learning.
            JK (str): The aggregation method for node representations.
            drop_ratio (float): The dropout rate after GNN layers.
            gnn_type (str): The type of GNN layer to use.
        """
        super(GINTrain, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, (self.emb_dim)//2),
            torch.nn.ELU(),
            torch.nn.Linear((self.emb_dim)//2, self.num_tasks))

    def super_node_rep(self, node_rep, batch):
        """
        Aggregates node representations to form super node representations.

        This method aggregates the node representations of each graph in the batch to form a super node
        representation by taking the representation of the last node for each graph in the batch.

        Args:
            node_rep (Tensor): The node representations of all nodes in the batch.
            batch (Tensor): The batch vector, which maps each node to its respective graph in the batch.

        Returns:
            Tensor: The super node representations for each graph in the batch.
        """
        super_group = []
        for i in range(len(batch)):
            if i != (len(batch)-1) and batch[i] != batch[i+1]:
                super_group.append(node_rep[i,:])
            elif i == (len(batch) -1):
                super_group.append(node_rep[i,:])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep

    def forward(self, *argv):
        """
        Forward pass of the GINTrain model.

        The method can accept either a data object or the components of the data object as separate parameters.
        It processes the input through GNN layers, aggregates the node representations to form super node
        representations, and applies a final prediction layer.

        Args:
            *argv: Variable length argument list. Can be a single data object or four separate components
                   of the data object (x, edge_index, edge_attr, batch).

        Returns:
            Tensor: The output predictions for each graph in the batch.
        """
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)
        
        return self.graph_pred_linear(super_rep)