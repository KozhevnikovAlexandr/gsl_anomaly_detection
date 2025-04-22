import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import argparse

from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator
from gnn import GNN_TAM


def parse_args():
    parser = argparse.ArgumentParser(description='model_inference')
    parser.add_argument('--dataset', type=str, default='reinartz_tep')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--name', type=str, default='gnn1')
    return parser.parse_args()


def inference():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Data preparation:
    dataset = FDDDataset(name=args.dataset)
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)
    test_dl = FDDDataloader(
        dataframe=dataset.df,
        label=dataset.label,
        mask=dataset.test_mask,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True
    )
    # Load saved model:
    model = torch.load('saved_models/' + args.name + '.pt',
                       map_location=device)
    # Inference:
    model.eval()
    preds = []
    test_labels = []
    for test_ts, test_index, test_label in test_dl:
        ts = torch.FloatTensor(test_ts).to(device)
        ts = torch.transpose(ts, 1, 2)
        with torch.no_grad():
            logits = model(ts)
        pred = logits.argmax(axis=1).cpu().numpy()
        preds.append(pd.Series(pred, index=test_index))
        test_labels.append(pd.Series(test_label, index=test_index))
    pred = pd.concat(preds)
    test_label = pd.concat(test_labels)
    
    evaluator = FDDEvaluator(step_size=1)
    evaluator.print_metrics(test_label, pred)


if __name__ == '__main__':
    inference()
