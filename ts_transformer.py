import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsl import GSL

class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(X)
        norm = adj.sum(1)**(-1/2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, theta=10000.0):
        super().__init__()
        self.d_head = d_model // nhead
        self.nhead = nhead
        self.theta = theta

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _apply_rope(self, Q, K):
        # Q, K: (B, nhead, T, d_head)
        B, H, T, D = Q.shape
        qr, qi = Q.view(B, H, T, D//2, 2).unbind(-1)
        kr, ki = K.view(B, H, T, D//2, 2).unbind(-1)
        freqs = torch.arange(0, D//2, device=Q.device) * (1.0 / self.theta)
        angles = torch.einsum('t,d->t d', torch.arange(T, device=Q.device), freqs)
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]
        qr2 = qr * cos - qi * sin
        qi2 = qr * sin + qi * cos
        kr2 = kr * cos - ki * sin
        ki2 = kr * sin + ki * cos
        Q2 = torch.stack([qr2, qi2], dim=-1).reshape(B, H, T, D)
        K2 = torch.stack([kr2, ki2], dim=-1).reshape(B, H, T, D)
        return Q2, K2

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, d_model = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # (B, nhead, T, d_head)
        Q = Q.view(B, T, self.nhead, self.d_head).transpose(1,2)
        K = K.view(B, T, self.nhead, self.d_head).transpose(1,2)
        V = V.view(B, T, self.nhead, self.d_head).transpose(1,2)
        Q, K = self._apply_rope(Q, K)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, T, d_model)
        return self.out_proj(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttentionRoPE(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        att = self.attn(x)
        x = self.norm1(x + self.dropout(att))
        ff = self.ff(x)
        return self.norm2(x + self.dropout(ff))

class TimeSeriesTransformerGSL(nn.Module):
    def __init__(self, ts_dim, window_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1,
                 gsl_k=None, gsl_alpha=1.0,
                 n_gnn=1, n_hidden=1024, device='cpu'):
        super().__init__()
        self.ts_dim = ts_dim
        self.window_size = window_size
        self.device = device
        self.layers = nn.ModuleList([
             TransformerBlock(d_model, nhead, dim_feedforward, dropout)
             for _ in range(num_layers)
         ])
        self.dropout = nn.Dropout(dropout)

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

        combined_dim = n_hidden + n_gnn * n_hidden
        self.fc = nn.Linear(combined_dim, combined_dim//2)
        self.fc_out = nn.Linear(combined_dim//2, ts_dim)

    def forward(self, x):
        # x: (B, T, N)
        B, T, N = x.size()
        x_proj = self.input_proj(x)
        for layer in self.layers:
             x_proj = layer(x_proj)
        
        transformer_feat = x_proj.mean(dim=1)  # (B, d_model)
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