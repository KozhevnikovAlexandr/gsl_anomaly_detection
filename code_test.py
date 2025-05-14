import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from fddbenchmark import FDDDataset
from dataset3 import ForecastFDDDataloader
import numpy as np
from tqdm.auto import tqdm, trange


def test(model, test_dl, scaler, dataset_obj, threshold=None, use_threshold_from_val=False):
    model.eval()
    all_errors = []
    all_true_labels = []

    with torch.no_grad():
        for i in range(len(test_dl) - 1):
            ts_batch, _, target_batch = test_dl[i]

            pred = model(ts_batch)

            target_raw = scaler.inverse_transform(target_batch.cpu().numpy())
            pred_raw = scaler.inverse_transform(pred.cpu().numpy())

            errors = np.abs(target_raw - pred_raw).mean(axis=1)
            all_errors.extend(errors)

            batch_idx = test_dl.batch_seq[i]
            batch_end_idx = test_dl.batch_seq[i + 1]
            batch_window_end_indices = test_dl.window_end_indices[batch_idx:batch_end_idx]
            target_indices = test_dl.index[batch_window_end_indices + 1]

            true_labels = (dataset_obj.label.loc[target_indices] != 0).astype(int).values
            all_true_labels.extend(true_labels)

    all_errors = np.array(all_errors)
    all_true_labels = np.array(all_true_labels)

    if len(all_errors) != len(all_true_labels):
        raise ValueError(f"Length mismatch: {len(all_errors)} errors vs {len(all_true_labels)} labels")

    if use_threshold_from_val or threshold is None:
        threshold = np.mean(all_errors) + 3 * np.std(all_errors)

    all_preds = (all_errors > threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(all_true_labels, all_preds),
        'precision': precision_score(all_true_labels, all_preds, zero_division=0),
        'recall': recall_score(all_true_labels, all_preds, zero_division=0),
        'f1': f1_score(all_true_labels, all_preds, zero_division=0),
    }

    print(f"Threshold used: {threshold:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    return metrics