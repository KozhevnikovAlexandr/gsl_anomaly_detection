import torch
import torch.nn as nn
import torch.nn.functional as F


# A = ReLu(W)
class Graph_ReLu_W(nn.Module):
    def __init__(self, n_nodes, k, device='cpu'):
        super(Graph_ReLu_W, self).__init__()
        self.num_nodes = n_nodes
        self.k = k
        self.device = device
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes).to(device),
                              requires_grad=True).to(device)

    def forward(self, idx):
        adj = F.relu(self.A)
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


# A for Directed graphs:
class Graph_Directed_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k, device):
        super(Graph_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k
        self.device = device
        self.e1 = nn.Embedding(n_nodes, window_size)
        self.e2 = nn.Embedding(n_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.l2 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha*torch.mm(m1, m2.transpose(1, 0))))
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


# A for Uni-directed graphs:
class Graph_Uni_Directed_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k, device):
        super(Graph_Uni_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k
        self.device = device
        self.e1 = nn.Embedding(n_nodes, window_size)
        self.e2 = nn.Embedding(n_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.l2 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha*(torch.mm(m1, m2.transpose(1, 0))
                                - torch.mm(m2, m1.transpose(1, 0)))))
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


# A for Undirected graphs:
class Graph_Undirected_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k, device):
        super(Graph_Undirected_A, self).__init__()
        self.alpha = alpha
        self.k = k
        self.device = device
        self.e1 = nn.Embedding(n_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        adj = F.relu(torch.tanh(self.alpha*torch.mm(m1, m2.transpose(1, 0))))
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


class GSL(nn.Module):
    """
    Graph structure learning block.
    """
    def __init__(
            self,
            gsl_type,
            n_nodes,
            window_size,
            alpha,
            k,
            device):
        super(GSL, self).__init__()
        self.gsl_layer = None
        if gsl_type == 'relu':
            self.gsl_layer = Graph_ReLu_W(n_nodes, k, device)
        elif gsl_type == 'directed':
            self.gsl_layer = Graph_Directed_A(n_nodes, window_size,
                                                   alpha, k, device)
        elif gsl_type == 'unidirected':
            self.gsl_layer = Graph_Uni_Directed_A(n_nodes, window_size,
                                                       alpha, k, device)
        elif gsl_type == 'undirected':
            self.gsl_layer = Graph_Undirected_A(n_nodes, window_size,
                                                     alpha, k, device)
        else:
            print('Wrong name of graph structure learning layer!')

    def forward(self, idx):
        return self.gsl_layer(idx)
