import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class Dataloader_Delhi_NCT(object):
    def __init__(self, data_path, flag, batch_size, num_workers, target):
        self.data_path = data_path
        self.flag = flag
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.read_data()

    def read_data(self):
        data = np.load(self.data_path)
        self.X = torch.tensor(data['X'], dtype=torch.float)
        self.y = torch.tensor(data['y'], dtype=torch.float)

    def get_dataloader(self):
        if self.flag == 'train':
            shuffle_flag = True
            drop_last = True
        elif self.flag in ('val', 'test'):
            shuffle_flag = False
            drop_last = False
        else:
            raise ValueError(f"未知的标志: {self.flag}")

        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=drop_last,
        )
        print(f"{self.flag} 数据准备完成 (Delhi NCT, shape X={tuple(self.X.shape)})")
        return dataloader

    def inverse_transform(self, data):
        scaler_path = os.path.join(
            os.path.dirname(self.data_path), f'scaler_{self.target}.npy'
        )
        try:
            scaler = np.load(scaler_path)
            vmin, vmax = scaler
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到 scaler 文件: {scaler_path}")
        except Exception as e:
            raise RuntimeError(f"无法加载 scaler 文件 {scaler_path}: {e}")

        return data * (vmax - vmin) + vmin
