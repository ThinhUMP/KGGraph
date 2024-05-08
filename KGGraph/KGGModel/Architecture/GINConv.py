import torch
from torch.nn import Linear, Parameter, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

num_bond_type = 9
num_bond_in_ring = 3
bond_type_1 = 2
bond_type_2 = 3
bond_type_3 = 2
bond_type_4 = 2
bond_type_5 = 6


class GINConv(MessagePassing):
    """
    GINConv is an extension of the Graph Isomorphism Network (GIN) that incorporates edge information
    by concatenating edge embeddings with node features before aggregation. This class extends the
    MessagePassing class to enable edge feature utilization in message passing.

    Args:
        dataset: The dataset object, which is used to derive the number of edge features or embeddings.
        emb_dim (int): The dimensionality of embeddings for nodes and edges.
        aggr (str): The aggregation scheme to use ('add', 'mean', 'max').

    Reference:
        Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?
        https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        """
        Initializes the GINConv layer with the specified parameters.

        Args:
            dataset: The dataset object, used for deriving the number of edge features.
            emb_dim (int): The dimensionality of node and edge embeddings.
            aggr (str): The aggregation method to use ('add', 'mean', 'max').
        """
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )

        self.edge_embedding0 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_in_ring, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(bond_type_1, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(bond_type_2, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(bond_type_3, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(bond_type_4, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding0.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GINConv layer.

        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
            edge_attr (Tensor): The edge attributes (features).

        Returns:
            Tensor: The updated node representations.
        """
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 8  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = (
            self.edge_embedding0(edge_attr[:, 0])
            + self.edge_embedding1(edge_attr[:, 1])
            + self.edge_embedding2(edge_attr[:, 2])
            + self.edge_embedding3(edge_attr[:, 3])
            + self.edge_embedding4(edge_attr[:, 4])
            + self.edge_embedding5(edge_attr[:, 5])
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """
        Constructs the messages to a node in a graph.

        Args:
            x_j (Tensor): The features of neighboring nodes.
            edge_attr (Tensor): The features of the edges.

        Returns:
            Tensor: The message to be aggregated.
        """
        return x_j + edge_attr

    def update(self, aggr_out):
        """
        Updates node features based on aggregated messages.

        Args:
            aggr_out (Tensor): The aggregated messages for each node.

        Returns:
            Tensor: The updated node features.
        """
        return self.mlp(aggr_out)
