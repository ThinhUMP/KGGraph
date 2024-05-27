import torch
import torch.nn.functional as F
from .GINConv import GINConv


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        JK="last",
        drop_ratio=0,
        gnn_type="gin",
        x_features=7,
        edge_features=5,
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # Initialize a list of x MLPs
        self.x_mlps = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(1, emb_dim),
                    # torch.nn.ReLU(),
                    # torch.nn.Linear(2 * emb_dim, emb_dim),
                )
                for _ in range(x_features)  # number of x features
            ]
        )

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr="add", edge_features=edge_features))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # Apply each MLP to its corresponding edge feature slice
        x_embeddings = torch.zeros(x.size(0), self.emb_dim).to(x.device)
        for i, mlp in enumerate(self.x_mlps):
            x_embeddings += mlp(x[:, i].view(-1, 1))

        h_list = [x_embeddings]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.elu(h), self.drop_ratio, training=self.training
                )  # relu->elu

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation
