import torch
import torch.nn as nn
import torch.nn.functional as F
from gsl import GSL

# class GraphConvolution(nn.Module):

#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
    
#     def forward(self, x, adj):
#         out = torch.matmul(adj, x)  # (B, N, in)
#         out = self.linear(out)      # (B, N, out)
#         return out

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
                 dim_feedforward=128, dropout=0.1, gsl_k=None, gsl_alpha=1.0, device='cpu'):
        super(TimeSeriesTransformerGSL, self).__init__()
        self.ts_dim = ts_dim
        self.window_size = window_size
        self.device = device
        
        self.input_proj = nn.Linear(ts_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.graph_proj = nn.Linear(ts_dim, d_model)

        self.fc = nn.Linear(d_model, d_model//2)
        self.fc_out = nn.Linear(d_model // 2, ts_dim)

        self.gsl = GSL(gsl_type='undirected', n_nodes=ts_dim, window_size=window_size, 
                       alpha=gsl_alpha, k=gsl_k, device=device)
        self.dropout = nn.Dropout(dropout*2)

    
    def forward(self, x):

        B, T, ts_dim = x.size()
        x_proj = self.input_proj(x)            # (B, window_size, d_model)
        x_proj = x_proj + self.pos_embedding 
        for layer in self.layers:
            x_proj = layer(x_proj)               # (B, window_size, d_model)
        transformer_feat = x_proj[:, -1, :]      # (B, d_model)
        
        idx = torch.arange(ts_dim, device=self.device)
        # (ts_dim, ts_dim)
        adj = self.gsl(idx)
        # (B, ts_dim)
        x_last = x[:, -1, :]
        graph_feat = torch.matmul(x_last, adj)   # (B, ts_dim)
        graph_feat = self.graph_proj(graph_feat)   # (B, d_model)

        transformer_feat = self.dropout(transformer_feat)
        graph_feat = self.dropout(graph_feat)

        combined = torch.cat((transformer_feat, graph_feat), dim=-1)  # (B, 2*d_model)
        out = self.fc(combined) # (B, d_model//2)
        out = F.sigmoid(out)
        out = self.fc_out(out) # (B, ts_dim)
        return out
