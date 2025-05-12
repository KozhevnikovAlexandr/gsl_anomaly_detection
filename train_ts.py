import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm, trange
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from fddbenchmark import FDDDataset, FDDDataloader
from dataset import ForecastFDDDataloader
from transformer import TimeSeriesTransformerGSL


def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer+GSL model for time series forecasting')
    parser.add_argument('--dataset', type=str, default='reinartz_tep')
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_gnn', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--gsl_k', type=int, default=5, help='Top-k для GSL')
    parser.add_argument('--name', type=str, default='transformer_gsl')
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    from fddbenchmark import FDDDataset
    dataset_obj = FDDDataset(name=args.dataset)
    
    scaler = StandardScaler()
    scaler.fit(dataset_obj.df[dataset_obj.train_mask])
    dataset_obj.df[:] = scaler.transform(dataset_obj.df)
    
    ts_dim = dataset_obj.df.shape[1] 
    train_dl = ForecastFDDDataloader(
        dataframe=dataset_obj.df,
        mask=dataset_obj.train_mask,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True,
        data_framework='torch',
        device=device
    )
    
    model = TimeSeriesTransformerGSL(ts_dim=ts_dim,
                                     window_size=args.window_size,
                                     d_model=args.d_model, # TODO поменять на ts_dim
                                     nhead=args.nhead,
                                     num_layers=args.num_layers,
                                     dim_feedforward=args.dim_feedforward,
                                     dropout=args.dropout,
                                     gsl_k=args.gsl_k,
                                     n_gnn=args.n_gnn,
                                    n_hidden=args.n_hidden,
                                     device=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    model.train()
    for epoch in trange(args.n_epochs, desc="Epochs"):
        losses = []
        for ts_batch, _, target_batch in tqdm(train_dl, desc="Batches", leave=False):
            optimizer.zero_grad()
            pred = model(ts_batch)  # (B, ts_dim)
            loss = F.l1_loss(pred, target_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{args.n_epochs}, MAE Loss: {np.mean(losses):.4f}")
    
    import os
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('saved_models', f'ts_{args.name}.pt'))

if __name__ == '__main__':
    train()