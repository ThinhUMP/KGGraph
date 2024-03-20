import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch
import torch.nn.functional as Fun
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
class gin(torch.nn.Module):
    """Graph Isomorphism Network class with 3 GINConv layers and 2 linear layers"""

    def __init__(self, dim_h, out_channels, dropout):
        """Initializing GIN class

        Args:
            dim_h (int): the dimension of hidden layers
        """
        self.dropout = dropout
        super(gin, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(126, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
                )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(dim_h, 2*dim_h), BatchNorm1d(2*dim_h), ReLU(), Linear(2*dim_h, 2*dim_h), ReLU(), Linear(2*dim_h, 2*dim_h), ReLU()
            )
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(2*dim_h, 4*dim_h), BatchNorm1d(4*dim_h), ReLU(), Linear(4*dim_h, 4*dim_h), ReLU(), Linear(4*dim_h, 4*dim_h), ReLU()
            )
        )
        self.conv4 = GINConv(
            Sequential(
                Linear(4*dim_h, 2*dim_h), BatchNorm1d(2*dim_h), ReLU(), Linear(2*dim_h, 2*dim_h), ReLU(), Linear(2*dim_h, 2*dim_h), ReLU()
            )
        )
        self.conv5 = GINConv(
            Sequential(
                Linear(2*dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
            )
        )
        self.lin1 = Linear(dim_h, 64)
        self.lin2 = Linear(64, out_channels)
        # self.act = torch.nn.Sigmoid()
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        # Node embeddings
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()
        h = self.conv4(h, edge_index)
        h = h.relu()
        h = self.conv5(h, edge_index)
        # Graph-level readout
        h = global_add_pool(h, batch)

        h = self.lin1(h)
        h = h.relu()
        h = Fun.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        # h = self.act(h)
        return h





