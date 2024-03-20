import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm, GIN

class GINNet(torch.nn.Module):
    def __init__(
        self, in_channels=126, hidden_channels=1024, num_layer=5, out_channels=1, dropout=0):
        self.num_layer = num_layer
        self.dropout = dropout
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        super().__init__()
        self.conv1 = GIN(in_channels, 2*hidden_channels, num_layers=num_layer)
        self.bn1 = BatchNorm(2*hidden_channels)
        self.conv2 = GIN(2*hidden_channels, 2*hidden_channels, num_layers=num_layer)
        self.bn2 = BatchNorm(2*hidden_channels)
        self.conv3 = GIN(2*hidden_channels, 4*hidden_channels, num_layers=num_layer)
        self.bn3 = BatchNorm(4*hidden_channels)
        self.conv4 = GIN(4*hidden_channels, 2*hidden_channels, num_layers=num_layer)
        self.bn4 = BatchNorm(2*hidden_channels)
        self.conv5 = GIN(2*hidden_channels, hidden_channels, num_layers=num_layer)
        self.bn5 = BatchNorm(hidden_channels)
        
        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.fc2 = torch.nn.Linear(64, out_channels)
        
        # self.act = torch.nn.Sigmoid()
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index 
        edge_attr = data.edge_attr.float()
        batch = data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index, edge_attr))
        x = self.bn5(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        # x = self.act(x)
        return x
