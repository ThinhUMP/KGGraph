import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add
#TODO: rewrite this python file to adapt with new code

vocab_edge_attr_embedding = 4

class GCNConv(MessagePassing):
    def __init__(self, dataset, in_channels, out_channels):
        super().__init__(aggr='mean')  # "Mean" aggregation (Step 5).
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.edge_embedding = torch.nn.Embedding(vocab_edge_attr_embedding, out_channels)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
        
    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype = dtype, device = edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

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
        
        edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)
        
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Compute normalization.
        norm = self.norm(edge_index, x.size(0), x.dtype)
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_attr = edge_embeddings,norm=norm)
        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, edge_attr, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * (x_j + edge_attr)