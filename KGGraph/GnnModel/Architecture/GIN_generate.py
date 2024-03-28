from .GINConv import GINConv
import torch
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

class GINGenerate(torch.nn.Module):
    def __init__(self, emb_dim, dropout, out_channels):
        super(GINGenerate, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.out_channels = out_channels
        # Using ModuleList to store each layer
        # self.convs = ModuleList()
        # for _ in range(num_layer):
        #     conv = GINConv(emb_dim)
        #     self.convs.append(conv)
        self.conv1 = GINConv(emb_dim=emb_dim)
        self.conv2 = GINConv(emb_dim=emb_dim)
        self.conv3 = GINConv(emb_dim=emb_dim)
        self.conv4 = GINConv(emb_dim=emb_dim)
        self.conv5 = GINConv(emb_dim=emb_dim)
        self.lin1 = Linear(emb_dim, emb_dim//2)
        self.lin2 = Linear(emb_dim//2, out_channels)
        
    def forward(self, data):
        
        x = data.x
        # print(x.size())
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        # print(len(batch.unique()))
        # for conv in self.convs:
        #     x = conv(x, edge_index, edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv5(x, edge_index, edge_attr)
        # print(x.size())
        x = global_add_pool(x, batch)
        # print(x.size())
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # print(x.size())
        
        return x
            