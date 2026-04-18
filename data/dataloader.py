"""
NPZ data loader shared across all DMGENet datasets.

NPZ layout (produced either by the raw-data builders or by
`data/compact_dataset.py` from the compact public station panels):
  X: (samples, N, seq_len, features)  float32
  y: (samples, N, pred_len)           float32

Scaler file (same directory):
  scaler_{target}.npy  — [min_val, max_val]; fit on the train split only.

Example:
  from data.dataloader import CityDataLoader
  loader = CityDataLoader(
      data_path='./dataset/Beijing_12/train_val_test_data/72_6/train_PM25.npz',
      flag='train', batch_size=64, num_workers=4, target='PM25'
  )
  train_loader = loader.get_dataloader()
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data.compact_dataset import ensure_prepared_splits


class CityDataLoader:
    """DataLoader wrapper for an NPZ-format air-quality dataset (any city)."""

    def __init__(self, data_path: str, flag: str,
                 batch_size: int, num_workers: int, target: str):
        """
        Args:
            data_path:   Path to the NPZ file.
            flag:        'train' | 'val' | 'test'
            batch_size:  Mini-batch size.
            num_workers: DataLoader worker processes.
            target:      Target variable (used to locate scaler_{target}.npy).
        """
        self.data_path   = data_path
        self.flag        = flag
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.target      = target
        self._read_data()

    def _read_data(self):
        if not os.path.exists(self.data_path):
            parts = self.data_path.replace("\\", "/").split("/")
            try:
                ds_idx = parts.index("dataset")
                dataset = parts[ds_idx + 1]
                seq_len, pred_len = parts[ds_idx + 3].split("_", 1)
                ensure_prepared_splits(dataset, horizons=[int(pred_len)], seq_len=int(seq_len))
            except Exception:
                pass
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "Run first: python data/compact_dataset.py --dataset <name>"
            )
        data = np.load(self.data_path)
        self.X = torch.tensor(data['X'], dtype=torch.float)
        self.y = torch.tensor(data['y'], dtype=torch.float)

    def get_dataloader(self) -> DataLoader:
        if self.flag == 'train':
            shuffle_flag = True
            drop_last    = True
        elif self.flag in ('val', 'test'):
            shuffle_flag = False
            drop_last    = False
        else:
            raise ValueError(f"Unknown flag: {self.flag!r}; expected 'train' | 'val' | 'test'")

        dataset    = TensorDataset(self.X, self.y)
        dataloader = DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = shuffle_flag,
            num_workers = self.num_workers,
            drop_last   = drop_last,
        )
        parts = self.data_path.replace("\\", "/").split("/")
        city_tag = parts[-4] if len(parts) >= 4 else "?"
        print(f"{self.flag} ready ({city_tag}, X={tuple(self.X.shape)}, y={tuple(self.y.shape)})")
        return dataloader

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Undo min-max scaling to recover real-valued PM2.5 in µg/m³."""
        scaler_path = os.path.join(
            os.path.dirname(self.data_path), f'scaler_{self.target}.npy'
        )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler file not found: {scaler_path}\n"
                "Ensure the dataset-build script produced scaler_{target}.npy"
            )
        scaler = np.load(scaler_path)
        vmin, vmax = float(scaler[0]), float(scaler[1])
        return data * (vmax - vmin) + vmin
