import torch
import torch.nn as nn
import torch.nn.functional as F
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_head = d_model // nhead
        self.nhead = nhead
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, d_model)
        B, T, d_model = x.size()
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        # (B, T, nhead, d_head) -> (B, nhead, T, d_head)
        Q = Q.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        # (B, nhead, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, nhead, T, d_head)
        # (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        out = self.out_proj(out)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TimeSeriesTransformerGSL(nn.Module):
    def __init__(self, ts_dim, window_size, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, dropout=0.1, 
                 gsl_k=None, gsl_alpha=1.0, 
                 n_gnn=1, n_hidden=1024, device='cpu'):
        super().__init__()
        self.ts_dim = ts_dim
        self.window_size = window_size
        self.device = device
        
        self.input_proj = nn.Linear(ts_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # GNN_TAM часть
        self.n_gnn = n_gnn
        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()
        self.z = (torch.ones(ts_dim, ts_dim) - torch.eye(ts_dim)).to(device)
        
        for _ in range(n_gnn):
            self.gsl.append(GSL(gsl_type='undirected', 
                                n_nodes=ts_dim, 
                                window_size=window_size, 
                                alpha=gsl_alpha, 
                                k=gsl_k, 
                                device=device))
            self.conv1.append(GCLayer(window_size, n_hidden))
            self.bnorm1.append(nn.BatchNorm1d(ts_dim))
            self.conv2.append(GCLayer(n_hidden, n_hidden))
            self.bnorm2.append(nn.BatchNorm1d(ts_dim))

        combined_dim = d_model + n_gnn * n_hidden
        self.fc = nn.Linear(combined_dim, combined_dim//2)
        self.fc_out = nn.Linear(combined_dim//2, ts_dim)

    def forward(self, x):
        B, T, N = x.size()
        
        x_proj = self.input_proj(x) + self.pos_embedding 
        for layer in self.layers:
            x_proj = layer(x_proj)
        transformer_feat = x_proj[:, -1, :]  # (B, d_model)
        transformer_feat = self.dropout(transformer_feat)

        x_gnn = x.transpose(1, 2)  # (B, N, T)
        gnn_features = []
        
        for i in range(self.n_gnn):
            adj = self.gsl[i](torch.arange(N).to(self.device)) * self.z
            h = self.conv1[i](adj, x_gnn).relu()
            h = self.bnorm1[i](h)
            skip, _ = torch.min(h, dim=1)
            h = self.conv2[i](adj, h).relu()
            h = self.bnorm2[i](h)
            h, _ = torch.min(h, dim=1)
            h += skip
            gnn_features.append(h)
        
        graph_feat = torch.cat(gnn_features, dim=1)  # (B, n_gnn*n_hidden)
        graph_feat = self.dropout(graph_feat)

        combined = torch.cat((transformer_feat, graph_feat), dim=-1)
        out = F.sigmoid(self.fc(combined))
        out = self.fc_out(out)
        return out
