import torch
from torch.nn import Linear, Parameter, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
# from ..vocab_edge_attr_embedding import num_vocab_edge_attr_embedding

vocab_edge_attr_embedding = 5
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
    def __init__(self, emb_dim, aggr = "add"):
        """
        Initializes the GINConv layer with the specified parameters.

        Args:
            dataset: The dataset object, used for deriving the number of edge features.
            emb_dim (int): The dimensionality of node and edge embeddings.
            aggr (str): The aggregation method to use ('add', 'mean', 'max').
        """
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            # torch.nn.BatchNorm1d(2*emb_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(2*emb_dim, 4*emb_dim),
            # torch.nn.BatchNorm1d(4*emb_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, emb_dim),
            )
        self.edge_embedding = torch.nn.Embedding(vocab_edge_attr_embedding, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.aggr = aggr
        self.in_channels = emb_dim
        self.emb_dim = emb_dim

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
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:,-5] = 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)

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
        aggr_out = torch.tensor(aggr_out, dtype=torch.float)
        return self.mlp(aggr_out)