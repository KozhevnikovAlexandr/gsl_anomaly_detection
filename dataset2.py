from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class ForecastingDataloader:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            mask: pd.Series,
            window_size: int,
            prediction_length: int = 1,
            dilation: int = 1,
            step_size: int = 1,
            use_minibatches: bool = False,
            batch_size: Optional[int] = None,
            shuffle: bool = False,
            random_state: Optional[int] = None,
            data_framework: str = 'numpy',
            device: str = 'cpu',
    ):
        if dataframe.index.names != ['run_id', 'sample']:
            raise ValueError("DataFrame must have multi-index ('run_id', 'sample')")
        if not np.all(dataframe.index == mask.index):
            raise ValueError("Dataframe and mask must have the same indices")
        
        self.df_values = dataframe.values
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.dilation = dilation
        self.step_size = step_size
        self.total_window_size = window_size + prediction_length
        self.data_framework = data_framework
        self.device = device

        # Генерация индексов с учетом временных сегментов
        window_end_indices = []
        run_ids = dataframe[mask].index.get_level_values(0).unique()
        
        for run_id in tqdm(run_ids, desc='Processing runs'):
            idx = dataframe.index.get_locs([run_id])
            mask_run = mask.iloc[idx]
            
            # Накапливаем непрерывные сегменты без аномалий
            valid_segments = []
            current_segment = []
            for i, valid in enumerate(mask_run):
                if valid:
                    current_segment.append(idx[i])
                else:
                    if len(current_segment) >= self.total_window_size:
                        valid_segments.append(current_segment)
                    current_segment = []
            if len(current_segment) >= self.total_window_size:
                valid_segments.append(current_segment)
            
            # Генерация окон внутри сегментов
            for segment in valid_segments:
                max_start = len(segment) - self.total_window_size
                if max_start < 0:
                    continue
                for start in range(0, max_start + 1, step_size):
                    end = start + self.total_window_size
                    window_end_indices.append(segment[end-1])

        # Перемешивание и батч-секвенции
        if shuffle:
            np.random.seed(random_state)
            window_end_indices = np.random.permutation(window_end_indices)
        else:
            window_end_indices = np.array(window_end_indices)
        
        batch_seq = np.arange(0, len(window_end_indices)+batch_size, batch_size) \
            if use_minibatches else np.array([0, len(window_end_indices)])
        self.batch_seq = batch_seq
        self.n_batches = len(batch_seq) - 1

        # Конвертация в фреймворк
        if data_framework == 'torch':
            import torch
            self.df_values = torch.tensor(self.df_values, device=device, dtype=torch.float32)
            self.batch_seq = torch.tensor(self.batch_seq, device=device)
            
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        end_indices = self.window_end_indices[self.batch_seq[idx]:self.batch_seq[idx+1]]
        seq_length = self.total_window_size * self.dilation
        start_indices = end_indices[:, None] - np.arange(0, seq_length, self.dilation)[::-1]
        
        # Разделение на X и y
        X = self.df_values[start_indices[:, :-self.prediction_length]]
        y = self.df_values[start_indices[:, -self.prediction_length:]]
        
        return X, y

class ForecastEvaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def evaluate(self, y_true, y_pred):
        # Регрессионные метрики
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        # Бинарные метрики
        errors = np.abs(np.array(y_true) - np.array(y_pred))
        anomaly_pred = (errors > self.threshold).astype(int)
        anomaly_true = (np.array(y_true) != 0).astype(int)  # Если есть метки
        
        precision = precision_score(anomaly_true, anomaly_pred)
        recall = recall_score(anomaly_true, anomaly_pred)
        f1 = f1_score(anomaly_true, anomaly_pred)
        roc_auc = roc_auc_score(anomaly_true, errors)
        
        return {
            'regression': {'MAE': mae, 'MSE': mse},
            'classification': {
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            }
        }
    
    def print_metrics(self, y_true, y_pred):
        metrics = self.evaluate(y_true, y_pred)
        print("Regression Metrics:")
        print(f"MAE: {metrics['regression']['MAE']:.4f}")
        print(f"MSE: {metrics['regression']['MSE']:.4f}")
        print("\nClassification Metrics:")
        print(f"Precision: {metrics['classification']['Precision']:.4f}")
        print(f"Recall: {metrics['classification']['Recall']:.4f}")
        print(f"F1-Score: {metrics['classification']['F1-Score']:.4f}")
        print(f"ROC-AUC: {metrics['classification']['ROC-AUC']:.4f}")
