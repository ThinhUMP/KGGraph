import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)
from torch_geometric.utils import normalized_cut
import torch_geometric.transforms as T
transform = T.Cartesian(cat=False)
def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nn1 = Sequential(
            Linear(19, 25),
            ReLU(),
            Linear(25, 126 * 32),
        )
        self.conv1 = NNConv(126, 32, nn1, aggr='mean')

        nn2 = Sequential(
            Linear(19, 25),
            ReLU(),
            Linear(25, 32 * 64),
        )
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        batch = data.batch
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        # weight = normalized_cut_2d(data.edge_index, data.pos)
        # cluster = graclus(data.edge_index, weight, data.x.size(0))
        # data.edge_attr = None
        # data = max_pool(cluster, data, transform=transform)

        x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        # weight = normalized_cut_2d(data.edge_index, data.pos)
        # cluster = graclus(data.edge_index, weight, data.x.size(0))
        # x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)