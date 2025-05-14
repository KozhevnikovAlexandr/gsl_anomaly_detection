import torch
import torch.nn as nn

from gsl import GSL


class GCLayer(nn.Module):
    """
    Graph convolution layer.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(X)
        norm = adj.sum(1)**(-1/2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h


class GNN_TAM(nn.Module):
    """
    Model architecture from the paper "Graph Neural Networks with Trainable
    Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data".
    https://doi.org/10.1109/ACCESS.2024.3481331
    """
    def __init__(
            self,
            n_nodes: int,
            window_size: int,
            n_classes: int,
            n_gnn: int = 1,
            gsl_type: str = 'relu',
            n_hidden: int = 1024,
            alpha: float = 0.1,
            k: int = None,
            device: str = 'cpu'
            ):
        """
        Args:
            n_nodes (int): The number of nodes/sensors.
            window_size (int): The number of timestamps in one sample.
            n_classes (int): The number of classes.
            n_gnn (int): The number of GNN modules.
            gsl_type (str): The type of GSL block.
            n_hidden (int): The number of hidden parameters in GCN layers.
            alpha (float): Saturation rate for GSL block.
            k (int): The maximum number of edges from one node.
            device (str): The name of a device to train the model. `cpu` and
                `cuda` are possible.
        """
        super(GNN_TAM, self).__init__()
        self.window_size = window_size
        self.nhidden = n_hidden
        self.device = device
        self.idx = torch.arange(n_nodes).to(device)
        self.adj = [0 for i in range(n_gnn)]
        self.h = [0 for i in range(n_gnn)]
        self.skip = [0 for i in range(n_gnn)]
        self.z = (torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)).to(device)
        self.n_gnn = n_gnn

        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for i in range(self.n_gnn):
            self.gsl.append(GSL(gsl_type, n_nodes,
                                window_size, alpha, k, device))
            self.conv1.append(GCLayer(window_size, n_hidden))
            self.bnorm1.append(nn.BatchNorm1d(n_nodes))
            self.conv2.append(GCLayer(n_hidden, n_hidden))
            self.bnorm2.append(nn.BatchNorm1d(n_nodes))

        self.fc = nn.Linear(n_gnn*n_hidden, n_classes)

    def forward(self, X):
        X = X.to(self.device)
        for i in range(self.n_gnn):
            self.adj[i] = self.gsl[i](self.idx)
            self.adj[i] = self.adj[i] * self.z
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])
            self.skip[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])
            self.h[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.h[i] + self.skip[i]

        h = torch.cat(self.h, 1)
        output = self.fc(h)

        return output

    def get_adj(self):
        return self.adj
