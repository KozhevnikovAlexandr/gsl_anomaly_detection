import torch
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np

class ForecastFDDDataloader:
    """
    Наследник FDDDataloader для прогнозирования следующего шага.
    В данном случае, вместо использования столбца label, целевая переменная 
    извлекается из dataframe – это значение следующего шага после окна.
    """
    def __init__(
            self,
            dataframe: pd.DataFrame,
            mask: pd.Series,
            window_size: int,
            dilation: int = 1,
            step_size: int = 1,
            use_minibatches: bool = False,
            batch_size: int = None,
            shuffle: bool = False,
            random_state: int = None,
            data_framework: str = 'numpy',
            device: str = 'cpu',
            disable_index: bool = False,
    ) -> None:
        self.dataframe = dataframe
        if dataframe.index.names != ['run_id', 'sample']:
            raise ValueError("``dataframe`` must have multi-index ('run_id', 'sample')")
        if not np.all(dataframe.index == mask.index):
            raise ValueError("``dataframe`` and ``mask`` must have the same indices.")

        if step_size > window_size:
            raise ValueError("``step_size`` must be less or equal to ``window_size``.")
        if use_minibatches and batch_size is None:
            raise ValueError("If you set ``use_minibatches=True``, "
                             "you must set ``batch_size`` to a positive number.")
        if data_framework not in ['numpy', 'torch']:
            raise ValueError("``data_framework`` must be in ('numpy', 'torch')")

        self.df_values = dataframe.values
        self.disable_index = disable_index
        if not disable_index:
            self.index = dataframe.index
        self.window_size = window_size
        self.dilation = dilation
        self.step_size = step_size

        window_end_indices = []
        run_ids = dataframe[mask].index.get_level_values(0).unique()
        for run_id in tqdm(run_ids, desc='Creating sequence of samples'):
            indices = np.array(dataframe.index.get_locs([run_id]))
            indices = indices[indices < (len(dataframe) - 1)]
            indices = indices[self.window_size - 1:]
            indices = indices[::step_size]
            indices = indices[mask.iloc[indices].to_numpy(dtype=bool)]
            window_end_indices.extend(indices)
        if random_state is not None:
            np.random.seed(random_state)
        self.window_end_indices = np.random.permutation(window_end_indices) if shuffle else np.array(window_end_indices)

        n_samples = len(self.window_end_indices)
        if use_minibatches:
            batch_seq = list(range(0, n_samples, batch_size))
            batch_seq.append(n_samples)
            self.batch_seq = np.array(batch_seq)
            self.n_batches = len(batch_seq) - 1
        else:
            self.batch_seq = np.array([0, n_samples])
            self.n_batches = 1

        if data_framework == 'torch':
            self.df_values = torch.tensor(self.df_values, device=device, dtype=torch.float32)
            self.batch_seq = torch.tensor(self.batch_seq, device=device, dtype=torch.long)
        self.device = device

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.n_batches:
            ts_batch, index_batch, target_batch = self.__getitem__(self.iter)
            self.iter += 1
            return ts_batch, index_batch, target_batch
        else:
            raise StopIteration

    def __getitem__(self, idx):
        ends_indices = self.window_end_indices[self.batch_seq[idx]:self.batch_seq[idx + 1]]
        windows_indices = ends_indices[:, None] - np.arange(0, self.window_size, self.dilation)[::-1]
        # (batch_size, window_size, ts_dim)
        ts_batch = self.df_values[windows_indices]
        target_indices = ends_indices + 1
        target_batch = self.df_values[target_indices]  # (batch_size, ts_dim)
        index_batch = None
        if not self.disable_index:
            index_batch = self.index[target_indices]
        return ts_batch, index_batch, target_batch