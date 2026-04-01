"""
Delhi NCT 单图基础模型实验模块
对标 exp_base_model.py，适配 Delhi 数据集路径与图构建逻辑。

差异:
  - 使用 data/dataloader_Delhi_NCT.py
  - 图矩阵在第一次需要时懒加载 (非模块级)，避免 import 时触发文件 I/O
  - 没有 POI 图 → 用功能相似代理图 (calculate_functional_matrix_delhi)
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model.model_1 import Model
from data.dataloader_Delhi_NCT import Dataloader_Delhi_NCT
from utils.metrics import metric_mutil_sites
from utils.tools import adjust_learning_rate, EarlyStopping, setup_seed

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── 懒加载图矩阵 ──────────────────────────────────────────────
_GRAPHS_DELHI = None


def _get_delhi_graphs():
    """首次调用时构建并缓存 Delhi 图矩阵"""
    global _GRAPHS_DELHI
    if _GRAPHS_DELHI is not None:
        return _GRAPHS_DELHI

    from _Support.Graph_Construction_Delhi import (
        calculate_distance_matrix_delhi,
        calculate_neighbor_matrix_delhi,
        calculate_similarity_matrix_delhi,
        calculate_functional_matrix_delhi,
    )

    def to_tensor(mat):
        return torch.tensor(mat, dtype=torch.float).to(DEVICE)

    adj_D, _, _ = calculate_distance_matrix_delhi(threshold=0.5, sigma=10.0)
    adj_N, _, _ = calculate_neighbor_matrix_delhi(radius_km=15.0)
    adj_S, _, _ = calculate_similarity_matrix_delhi(threshold=0.6, target='PM25')
    adj_F, _, _ = calculate_functional_matrix_delhi(threshold=0.7)

    _GRAPHS_DELHI = {
        'Model_D':   to_tensor(adj_D),
        'Model_N':   to_tensor(adj_N),
        'Model_S':   to_tensor(adj_S),
        'Model_POI': to_tensor(adj_F),   # 使用功能代理图替代 POI
    }
    return _GRAPHS_DELHI


class Exp_model_Delhi:
    def __init__(
        self, model_name, model, epoch, learning_rate, target,
        batch_size, num_workers, seq_len, predict_len,
        results_folder_override=None,
    ):
        self.model_name = model_name
        self.model = model.to(DEVICE)
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.target = target

        dataset = 'Delhi_NCT'
        print(f'dataset:{dataset}  model:{self.model_name}  target:{self.target}  predict_len:{predict_len}')

        if results_folder_override is not None:
            self.results_folder = results_folder_override
        else:
            self.results_folder = os.path.join(
                f'./预测结果_基础模型_{dataset}', str(seq_len), str(predict_len), self.model_name
            )
        os.makedirs(self.results_folder, exist_ok=True)

        root_path = f'./dataset/{dataset}/train_val_test_data/{seq_len}_{predict_len}'
        self.train_dataloader = Dataloader_Delhi_NCT(
            os.path.join(root_path, f'train_{self.target}.npz'), 'train', batch_size, num_workers, target)
        self.val_dataloader   = Dataloader_Delhi_NCT(
            os.path.join(root_path, f'val_{self.target}.npz'),   'val',   batch_size, num_workers, target)
        self.test_dataloader  = Dataloader_Delhi_NCT(
            os.path.join(root_path, f'test_{self.target}.npz'),  'test',  batch_size, num_workers, target)
        self.train_loader = self.train_dataloader.get_dataloader()
        self.val_loader   = self.val_dataloader.get_dataloader()
        self.test_loader  = self.test_dataloader.get_dataloader()

    # ── internal helpers ────────────────────────────────────────

    def val(self, criterion):
        val_loss, test_loss = [], []
        self.model.eval()
        with torch.no_grad():
            for features, target in self.val_loader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                loss = criterion(self.model(features), target)
                val_loss.append(loss.item())
            for features, target in self.test_loader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                loss = criterion(self.model(features), target)
                test_loss.append(loss.item())
        self.model.train()
        return np.average(val_loss), np.average(test_loss)

    # ── train ───────────────────────────────────────────────────

    def train(self):
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses, val_losses, test_losses, epoch_times = [], [], [], []

        self.model.train()
        t0 = time.time()
        early_stopping = EarlyStopping(
            patience=7, verbose=True,
            path=os.path.join(self.results_folder, self.model_name + '.pth')
        )

        for epoch in range(self.epoch):
            epoch_train_loss = []
            t_ep = time.time()
            for features, target in self.train_loader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                optim.zero_grad()
                loss = criterion(self.model(features), target)
                epoch_train_loss.append(loss.item())
                loss.backward()
                optim.step()

            ep_time = time.time() - t_ep
            epoch_times.append(ep_time)

            ep_train = np.average(epoch_train_loss)
            ep_val, ep_test = self.val(criterion)
            train_losses.append(ep_train)
            val_losses.append(ep_val)
            test_losses.append(ep_test)

            print(
                f"Epoch [{epoch+1:3d}/{self.epoch}]  time:{ep_time:.1f}s  "
                f"train:{ep_train:.5f}  val:{ep_val:.5f}  test:{ep_test:.5f}",
                end=' '
            )
            adjust_learning_rate(optim, epoch + 1, self.learning_rate)
            early_stopping(ep_val, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        elapsed = time.time() - t0
        print(f"训练时间 {elapsed:.1f}s ({elapsed/60:.1f}min)")

        pd.DataFrame({
            'epoch_time': epoch_times,
            'train_loss': train_losses,
            'val_loss':   val_losses,
            'test_loss':  test_losses,
        }).to_csv(os.path.join(self.results_folder, 'loss.csv'), index=True, index_label='epoch')


        best_path = os.path.join(self.results_folder, self.model_name + '.pth')
        self.model.load_state_dict(torch.load(best_path))

    # ── evaluate ────────────────────────────────────────────────

    def _evaluate(self, dataloader, loader, flag):
        predictions, trues, features_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for features, target in loader:
                features_list.append(features.numpy())
                pred = self.model(features.to(DEVICE))
                predictions.append(pred.cpu().numpy())
                trues.append(target.numpy())

        all_features  = np.concatenate(features_list, axis=0)
        trues          = np.concatenate(trues,        axis=0)
        predictions    = np.concatenate(predictions,  axis=0)

        print(f'{flag}: features{all_features.shape}  pred{predictions.shape}  true{trues.shape}')

        np.save(os.path.join(self.results_folder, f'{flag}_X.npy'),           all_features)
        np.save(os.path.join(self.results_folder, f'{flag}_y.npy'),           trues)
        np.save(os.path.join(self.results_folder, f'{flag}_predictions.npy'), predictions)

        trues_inv = dataloader.inverse_transform(trues)
        preds_inv = dataloader.inverse_transform(predictions)

        np.save(os.path.join(self.results_folder, f'{flag}_y_inverse.npy'),           trues_inv)
        np.save(os.path.join(self.results_folder, f'{flag}_predictions_inverse.npy'), preds_inv)

        metrics = metric_mutil_sites(preds_inv, trues_inv)
        print(f'{flag}  MAE:{metrics[0]:.3f}  RMSE:{metrics[1]:.3f}  IA:{metrics[2]:.4f}  R2:{metrics[3]:.3f}')

        pd.DataFrame([metrics], columns=[f'{flag}_MAE', f'{flag}_RMSE', f'{flag}_IA', f'{flag}_R2']) \
          .to_csv(os.path.join(self.results_folder, f'{flag}_metrics.csv'), index=False)

    def test(self):
        self.model.eval()
        self._evaluate(self.val_dataloader,  self.val_loader,  'val')
        self._evaluate(self.test_dataloader, self.test_loader, 'test')


# ─── 独立运行入口 ──────────────────────────────────────────────
if __name__ == '__main__':
    IN_CHANNELS  = 12
    SEQ_LEN      = 72
    NUM_NODES    = 12
    HIDDEN_SIZE  = 64
    DROPOUT      = 0.2
    BLOCK_NUM    = 2
    NUM_HEADS    = 4
    ALPHA        = 0.2
    APT_SIZE     = 10
    NUM_CHANNELS = [64, 64, 64, 64]
    EPOCH        = 100
    LR           = 0.001
    BATCH_SIZE   = 64
    NUM_WORKERS  = 3

    setup_seed(2026)
    graphs = _get_delhi_graphs()

    for T_out in [1, 6, 12, 24]:
        for model_name, adj in graphs.items():
            model = Model(
                adj, IN_CHANNELS, HIDDEN_SIZE, DROPOUT, ALPHA, NUM_HEADS,
                num_channels=NUM_CHANNELS, apt_size=APT_SIZE,
                num_nodes=NUM_NODES, num_block=BLOCK_NUM,
                T_in=SEQ_LEN, predict_len=T_out,
                gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True,
            )
            exp = Exp_model_Delhi(
                model_name, model, EPOCH, LR, 'PM25',
                BATCH_SIZE, NUM_WORKERS, SEQ_LEN, T_out,
            )
            print(f"{model_name} 训练开始")
            exp.train()
            exp.test()
            print(f"{model_name} 完成\n{'='*80}")
            torch.cuda.empty_cache()
