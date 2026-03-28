import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class Dataloader_Beijing_12(object):
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
        elif self.flag == 'val':
            shuffle_flag = False
            drop_last = False
        elif self.flag == 'test':
            shuffle_flag = False
            drop_last = False
        else:
            raise ValueError(f"未知的标志: {self.flag}")

        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle_flag,
                                num_workers=self.num_workers,
                                drop_last=drop_last)
        print(f"{self.flag} 数据准备完成")
        return dataloader

    def inverse_transform(self, data):
        if self.target == 'PM25':
            scaler_path = os.path.join(os.path.dirname(self.data_path), 'scaler_PM25.npy')
        elif self.target == 'PM10':
            scaler_path = os.path.join(os.path.dirname(self.data_path), 'scaler_PM10.npy')
        else:
            raise ValueError(f"未知的目标变量: {self.target}")

        try:
            scaler = np.load(scaler_path)
            # mean, std = scaler
            min, max = scaler
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {scaler_path} 不存在")
        except Exception as e:
            raise RuntimeError(f"无法加载文件 {scaler_path}: {e}")

        # data_inverse = data * std + mean
        data_inverse = data * (max - min) + min
        return data_inverse