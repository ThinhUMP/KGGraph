import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2*hidden_channels)
        self.conv2 = GCNConv(2*hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.act = torch.nn.Sigmoid()
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        x = F.dropout(x, p=0, training=self.training)
        x = self.conv1(x, edge_index, edge_attr).relu()
        print(x.shape)
        x = F.dropout(x, p=0, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        print(x.shape)
        x = self.lin(x)
        print(x.shape)
        x = self.act(x)
        print(x.shape)
        return x