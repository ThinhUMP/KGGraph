import torch
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn import global_add_pool
import torch
import torch.nn.functional as Fun
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
class RGGraph(torch.nn.Module):
    """Graph Isomorphism Network class with 3 GINConv layers and 2 linear layers"""

    def __init__(self, dim_h):
        """Initializing GIN class

        Args:
            dim_h (int): the dimension of hidden layers
        """
        super(RGGraph, self).__init__()
        self.conv1 = ResGatedGraphConv(126, dim_h)
        self.conv2 = ResGatedGraphConv(dim_h, dim_h)
        self.conv3 = ResGatedGraphConv(dim_h, dim_h)
        
        self.lin1 = Linear(dim_h, 64)
        self.lin2 = Linear(64, 1)
        self.act = torch.nn.Sigmoid()
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        # Node embeddings
        h = self.conv1(x, edge_index, edge_attr)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_attr)
        h = h.relu()
        h = self.conv3(h, edge_index, edge_attr)

        # Graph-level readout
        h = global_add_pool(h, batch)

        h = self.lin1(h)
        h = h.relu()
        h = Fun.dropout(h, p=0, training=self.training)
        h = self.lin2(h)
        h = self.act(h)
        return h