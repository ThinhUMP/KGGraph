from .GINConv import GINConv
import torch
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

class GINGenerate(torch.nn.Module):
    def __init__(self, in_channels, emb_dim, dropout, out_channels):
        super(GINGenerate, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.out_channels = out_channels
        # Using ModuleList to store each layer
        # self.convs = ModuleList()
        # for _ in range(num_layer):
        #     conv = GINConv(emb_dim)
        #     self.convs.append(conv)
        self.conv1 = GINConv(in_channels= in_channels, emb_dim=emb_dim)
        self.conv2 = GINConv(in_channels= 2*emb_dim, emb_dim=4*emb_dim)
        # self.conv3 = GINConv(in_channels= 2*emb_dim, emb_dim=4*emb_dim)
        self.lin1 = Linear(8*emb_dim, 128)
        self.lin2 = Linear(128, out_channels)
        
    def forward(self, data):
        
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        # for conv in self.convs:
        #     x = conv(x, edge_index, edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv3(x, edge_index, edge_attr)
        
        x = global_add_pool(x, batch)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x
            