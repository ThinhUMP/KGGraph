from .GINConv import GINConv
import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import global_add_pool
import torch.functional as F

class GINGenerate(torch.nn.Module):
    def __init__(self, emb_dim, dropout, num_layer):
        super(GINGenerate, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_layer = num_layer
        # Using ModuleList to store each layer
        self.convs = ModuleList()
        for _ in range(num_layer):
            conv = GINConv(emb_dim)
            self.convs.append(conv)
            
        self.lin1 = Linear(emb_dim, 64)
        self.lin2 = Linear(64, 1)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
            