import torch
import numpy as np
import pandas as pd

class ForecastSimpleDataloader:
    """
    Даталоадер для прогнозирования следующего шага многомерного временного ряда.
    На вход принимается DataFrame с колонкой меток Normal/Attack (0/1).
    При train режимe (train=True) исключает любые окна, содержащие класс Attack.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        window_size: int,
        dilation: int = 1,
        step_size: int = 1,
        use_minibatches: bool = False,
        batch_size: int = None,
        shuffle: bool = False,
        data_framework: str = 'numpy',
        device: str = 'cpu',
        train: bool = True,
        timestamp_col: str = 'Timestamp',
        label_col: str = 'Normal/Attack',
        disable_index: bool = False,
        random_state: int = None
    ) -> None:
        # Проверки
        if step_size <= 0:
            raise ValueError("`step_size` must be > 0.")
        if dilation <= 0:
            raise ValueError("`dilation` must be > 0.")
        if use_minibatches and (batch_size is None or batch_size <= 0):
            raise ValueError("`batch_size` must be set to a positive integer when use_minibatches=True.")
        if data_framework not in ['numpy', 'torch']:
            raise ValueError("`data_framework` must be 'numpy' or 'torch'.")
        
        # Сохраняем параметры
        self.window_size = window_size
        self.dilation = dilation
        self.step_size = step_size
        self.disable_index = disable_index
        self.train = train
        self.device = device
        
        # Копируем и подготавливаем данные
        df = dataframe.copy()
        # Обрабатываем колонки
        if timestamp_col in df.columns:
            self.timestamps = df[timestamp_col].values
            df = df.drop(columns=[timestamp_col])
        else:
            self.timestamps = np.arange(len(df))
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
        labels = df[label_col].values.astype(int)
        self.labels = labels
        # Маскирование: при train исключаем все точки с меткой 1
        mask = (labels == 0) if train else np.ones_like(labels, dtype=bool)
        # Убираем колонку меток из признаков
        df = df.drop(columns=[label_col])
        self.feature_values = df.values
        self.n_samples_total = len(df)
        self.n_features = df.shape[1]
        
        # Генерация окон
        ends = []
        # Итерация по возможным окончаниям окна
        for end in range((window_size - 1) * dilation, self.n_samples_total - 1, step_size):
            # Индексы окна
            window_idx = end - dilation * np.arange(window_size)[::-1]
            # Проверка выхода за границы
            if window_idx.min() < 0:
                continue
            # Проверка маски (окно и точка таргета)
            if not mask[window_idx].all():
                continue
            if not mask[end + 1]:
                continue
            ends.append(end)
        if shuffle:
            rng = np.random.RandomState(seed=random_state)
            rng.shuffle(ends)
        self.window_ends = np.array(ends, dtype=int)
        self.n_windows = len(self.window_ends)
        
        # Batch sequence
        if use_minibatches:
            batch_seq = list(range(0, self.n_windows, batch_size)) + [self.n_windows]
        else:
            batch_seq = [0, self.n_windows]
        self.batch_seq = np.array(batch_seq, dtype=int)
        self.n_batches = len(self.batch_seq) - 1
        
        # Преобразование в torch.tensor, если нужно
        if data_framework == 'torch':
            self.feature_values = torch.tensor(self.feature_values, dtype=torch.float32, device=device)
            self.batch_seq = torch.tensor(self.batch_seq, dtype=torch.long, device=device)
            self.window_ends = torch.tensor(self.window_ends, dtype=torch.long, device=device)
        
    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= self.n_batches:
            raise StopIteration
        batch = self._iter
        result = self.__getitem__(batch)
        self._iter += 1
        return result

    def __getitem__(self, idx):
        start, end = self.batch_seq[idx], self.batch_seq[idx + 1]
        ends = self.window_ends[start:end]
        
        if isinstance(ends, (torch.Tensor,)):
            ends_cpu = ends.cpu().numpy()
        else:
            ends_cpu = ends
        windows_idx = ends_cpu[:, None] - self.dilation * np.arange(self.window_size)[::-1]
        
        ts_batch = self.feature_values[windows_idx]
        
        target_idx = ends_cpu + 1
        target_batch = self.feature_values[target_idx]
        labels_batch = self.labels[target_idx] if hasattr(self, 'labels') else None
        
        index_batch = None
        if not self.disable_index:
            index_batch = self.timestamps[target_idx]
        
        return ts_batch, index_batch, target_batch, labels_batch