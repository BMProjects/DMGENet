"""
噪声鲁棒性实验 — 在测试数据中注入不同水平高斯噪声
回应 Reviewer #2 Comment 12: 鲁棒性测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from model.model_1 import Model
from data.dataloader_Beijing_12 import Dataloader_Beijing_12
from utils.metrics import metric_mutil_sites as metric
from _Support.Graph_Construction_Beijing_12 import (
    calculate_the_distance_matrix,
    calculate_the_neighbor_matrix,
    calculate_the_similarity_matrix,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]  # 0%, 5%, 10%, 15%, 20%
HORIZONS = [1, 6, 12, 24]
DATASET = 'Beijing_12'
SEQ_LEN = 72
BATCH_SIZE = 64

# 图矩阵
print("构建图矩阵...")
adj_D, _, _ = calculate_the_distance_matrix(threshold=0.4)
adj_D = torch.tensor(adj_D, dtype=torch.float).to(DEVICE)
adj_N, _, _ = calculate_the_neighbor_matrix(45)
adj_N = torch.tensor(adj_N, dtype=torch.float).to(DEVICE)
adj_S, _, _ = calculate_the_similarity_matrix(threshold=0.6, target='PM25')
adj_S = torch.tensor(adj_S, dtype=torch.float).to(DEVICE)
adj_POI = pd.read_csv('./dataset/Beijing_12/POI/adjacency_matrix.csv', header=None)
adj_POI = torch.tensor(adj_POI.values, dtype=torch.float).to(DEVICE)
adj_POI = torch.where(adj_POI >= 0.7, adj_POI, torch.zeros_like(adj_POI))

GRAPH_MODELS = {
    'Model_D': adj_D, 'Model_N': adj_N,
    'Model_S': adj_S, 'Model_POI': adj_POI,
}


def add_noise(data_tensor, noise_level):
    """在 min-max 归一化后的数据上注入高斯噪声"""
    if noise_level == 0:
        return data_tensor
    noise = torch.randn_like(data_tensor) * noise_level
    return data_tensor + noise


def evaluate_with_noise(model, dataloader, scaler, noise_level, predict_len):
    model.eval()
    preds, trues = [], []
    loader = dataloader.get_dataloader()

    with torch.no_grad():
        for features, target in loader:
            features = features.float().to(DEVICE)
            target = target.float().to(DEVICE)

            # 注入噪声到输入特征
            features_noisy = add_noise(features, noise_level)
            output = model(features_noisy)

            preds.append(output.cpu().numpy())
            trues.append(target.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 反归一化 (min-max)
    _min, _max = scaler[0], scaler[1]
    preds_inv = preds * (_max - _min) + _min
    trues_inv = trues * (_max - _min) + _min

    mae, rmse, ia, r2 = metric(preds_inv, trues_inv)
    return {'MAE': mae, 'RMSE': rmse, 'IA': ia, 'R2': r2}


def main():
    results = []

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"Horizon: {horizon}h")
        print(f"{'='*60}")

        root_path = f'./dataset/Beijing_12/train_val_test_data/{SEQ_LEN}_{horizon}'
        scaler = np.load(os.path.join(root_path, 'scaler_PM25.npy'))

        test_dl = Dataloader_Beijing_12(
            os.path.join(root_path, 'test_PM25.npz'),
            'test', BATCH_SIZE, 0, 'PM25'
        )

        for model_name, adj in GRAPH_MODELS.items():
            # 加载已训练的最佳模型
            ckpt_dir = f'./预测结果_基础模型_{DATASET}/{SEQ_LEN}/{horizon}/{model_name}'
            ckpt_path = os.path.join(ckpt_dir, f'{model_name}.pth')

            if not os.path.exists(ckpt_path):
                print(f"  跳过 {model_name} h={horizon}: 没有找到 {ckpt_path}")
                continue

            model = Model(
                adj, 12, 64, 0.2, 0.2, 4,
                num_channels=[64, 64, 64, 64], apt_size=10,
                num_nodes=12, num_block=2,
                T_in=SEQ_LEN, predict_len=horizon,
                gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True,
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

            for noise_level in NOISE_LEVELS:
                metrics = evaluate_with_noise(model, test_dl, scaler, noise_level, horizon)
                row = {
                    'horizon': horizon,
                    'model': model_name,
                    'noise_level': noise_level,
                    **metrics
                }
                results.append(row)
                print(f"  {model_name} noise={noise_level:.0%}: "
                      f"RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} IA={metrics['IA']:.4f}")

            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    out_path = './doc/noise_robustness.csv'
    df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

    # 打印论文摘要
    print("\n" + "=" * 80)
    print("论文 Table: Noise Robustness (avg across 4 graph models)")
    print("=" * 80)
    summary = df.groupby(['horizon', 'noise_level']).agg({'RMSE': 'mean', 'MAE': 'mean', 'IA': 'mean'}).reset_index()
    for h in HORIZONS:
        print(f"\n--- {h}h ---")
        sub = summary[summary['horizon'] == h]
        baseline_rmse = sub[sub['noise_level'] == 0.0]['RMSE'].values[0]
        for _, r in sub.iterrows():
            delta = (r['RMSE'] - baseline_rmse) / baseline_rmse * 100
            print(f"  noise={r['noise_level']:.0%}: RMSE={r['RMSE']:.2f} (+{delta:.1f}%)  MAE={r['MAE']:.2f}  IA={r['IA']:.4f}")


if __name__ == '__main__':
    main()
