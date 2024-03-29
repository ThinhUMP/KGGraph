import torch
from torch.nn import Linear, Parameter, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from ..vocab_edge_attr_embedding import num_vocab_edge_attr_embedding

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, dataset, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            torch.nn.BatchNorm1d(2*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, 4*emb_dim),
            torch.nn.BatchNorm1d(4*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*emb_dim, emb_dim),
            )
        vocab_edge_attr_embedding = num_vocab_edge_attr_embedding(dataset)
        self.edge_embedding = torch.nn.Embedding(vocab_edge_attr_embedding, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.aggr = aggr
        self.in_channels = emb_dim
        self.emb_dim = emb_dim

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:,-4] = 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        aggr_out = torch.tensor(aggr_out, dtype=torch.float)
        return self.mlp(aggr_out)