from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm, trange
import argparse

from fddbenchmark import FDDDataset, FDDDataloader
from dataset import ForecastFDDDataloader
from gnn import GNN_TAM


def parse_args():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--dataset', type=str, default='reinartz_tep')
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_gnn', type=int, default=1)
    parser.add_argument('--gsl_type', type=str, default='relu')
    parser.add_argument('--n_hidden', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--name', type=str, default='gnn')
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Data preparation:
    dataset = FDDDataset(name=args.dataset)
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)
    n_nodes = dataset.df.shape[1]
    n_classes = len(set(dataset.label))
    train_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.train_mask,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True
    )
    # Model creation:
    model = GNN_TAM(n_nodes=n_nodes,
                    window_size=args.window_size,
                    n_classes=n_classes,
                    n_gnn=args.n_gnn,
                    gsl_type=args.gsl_type,
                    n_hidden=args.n_hidden,
                    alpha=args.alpha,
                    k=args.k,
                    device=device)
    model.to(device)
    # Training:
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    weight = torch.ones(n_classes) * 0.5
    weight[1:] /= (n_classes - 1)
    for e in trange(args.n_epochs, desc="Epochs ..."):
        av_loss = []
        for train_ts, train_index, train_label in tqdm(train_dl):
            ts = torch.FloatTensor(train_ts).to(device)
            ts = torch.transpose(ts, 1, 2)
            train_label = torch.LongTensor(train_label).to(device)
            logits = model(ts)
            loss = F.cross_entropy(logits, train_label, weight=weight.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            av_loss.append(loss.item())
        print(f'Epoch: {e+1:2d}/{args.n_epochs}, average CE loss: {sum(av_loss)/len(av_loss):.4f}')

    torch.save(model, 'saved_models/' + args.name + str(args.n_gnn) + '.pt')


if __name__ == '__main__':
    train()
