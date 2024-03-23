import torch
from torch.nn import Linear, Parameter, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add

unique_value_edge_attr = 2

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr='Add')  # "Add" aggregation (Step 3).
        self.emb_dim = emb_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            BatchNorm1d(2*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            )
        self.bias = Parameter(torch.empty(emb_dim))
        self.edge_embedding = torch.nn.Embedding(unique_value_edge_attr, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        
    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:,-4] = 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        
        edge_embeddings = torch.zeros(edge_attr.size(0), self.emb_dim).to(edge_attr.device).to(edge_attr.dtype)

        for i in range(edge_attr.size(1)):  # Iterate over the second dimension
            embedding_ith = self.edge_embedding(edge_attr[:, i]).clone().detach().to(edge_attr.device).to(edge_attr.dtype)
            edge_embeddings += embedding_ith
        
        # Step 2-3: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_attr = edge_embeddings)
        # Step56: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]
        return x_j + edge_attr
    
    def update(self, aggr_out):
        return self.mlp(aggr_out)