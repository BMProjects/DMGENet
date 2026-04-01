"""
多随机种子基础模型训练 — 为统计显著性检验提供方差数据
回应 Reviewer #2 Comment 10: 统计显著性

仅训练 4 个主模型 (D/N/S/POI), 每个种子各自独立训练.
结果保存到 预测结果_基础模型_Beijing_12_multiseed/{seed}/{seq_len}/{predict_len}/{model}/
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from model.model_1 import Model
from exp_base_model import Exp_model
from utils.tools import setup_seed
from _Support.Graph_Construction_Beijing_12 import (
    calculate_the_distance_matrix,
    calculate_the_neighbor_matrix,
    calculate_the_similarity_matrix,
)

# ─── 参数 ──────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 2024]   # 5 个独立种子
HORIZONS = [1, 6, 12, 24]
DATASET = 'Beijing_12'
SEQ_LEN = 72
BATCH_SIZE = 64
EPOCH = 100
LR = 0.001
NUM_WORKERS = 3

IN_CHANNELS = 12
HIDDEN_SIZE = 64
DROPOUT = 0.2
BLOCK_NUM = 2
NUM_HEADS = 4
ALPHA = 0.2
APT_SIZE = 10
NUM_CHANNELS = [64, 64, 64, 64]
NUM_NODES = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_ROOT = f'./预测结果_基础模型_{DATASET}_multiseed'

# ─── 图矩阵 ────────────────────────────────────────────────────
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

ADJ_MAP = {
    'Model_D': adj_D, 'Model_N': adj_N,
    'Model_S': adj_S, 'Model_POI': adj_POI,
}

results_all = []


def make_model(adj, T_out):
    return Model(
        adj, IN_CHANNELS, HIDDEN_SIZE, DROPOUT, ALPHA, NUM_HEADS,
        num_channels=NUM_CHANNELS, apt_size=APT_SIZE,
        num_nodes=NUM_NODES, num_block=BLOCK_NUM,
        T_in=SEQ_LEN, predict_len=T_out,
        gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True,
    )


def run_one(seed, T_out, model_name, adj):
    setup_seed(seed)
    save_dir = os.path.join(OUTPUT_ROOT, f'seed_{seed}', str(SEQ_LEN), str(T_out), model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 跳过已完成
    if os.path.exists(os.path.join(save_dir, 'test_metrics.csv')):
        df = pd.read_csv(os.path.join(save_dir, 'test_metrics.csv'))
        row = df.iloc[0].to_dict()
        row.update({'seed': seed, 'horizon': T_out, 'model': model_name})
        print(f"  [跳过] seed={seed} h={T_out} {model_name}: RMSE={row.get('test_RMSE', '?'):.4f}")
        return row

    model = make_model(adj, T_out)
    exp = Exp_model(
        model_name, model, EPOCH, LR, 'PM25',
        BATCH_SIZE, NUM_WORKERS, DATASET, SEQ_LEN, T_out,
        results_folder_override=save_dir,
    )
    exp.train()
    exp.test()

    metrics = pd.read_csv(os.path.join(save_dir, 'test_metrics.csv')).iloc[0].to_dict()
    metrics.update({'seed': seed, 'horizon': T_out, 'model': model_name})
    print(f"  seed={seed} h={T_out} {model_name}: RMSE={metrics.get('test_RMSE', '?'):.4f} MAE={metrics.get('test_MAE', '?'):.4f}")
    return metrics


def main():
    for seed in SEEDS:
        for T_out in HORIZONS:
            print(f"\n{'='*60}")
            print(f"Seed={seed}, Horizon={T_out}h")
            print(f"{'='*60}")
            for model_name, adj in ADJ_MAP.items():
                row = run_one(seed, T_out, model_name, adj)
                results_all.append(row)
                torch.cuda.empty_cache()

    # 汇总统计
    df = pd.DataFrame(results_all)
    df.to_csv('./doc/multiseed_results.csv', index=False)

    print("\n" + "="*70)
    print("多种子统计汇总 (mean ± std, 5 seeds × 4 models)")
    print("="*70)
    for h in HORIZONS:
        sub = df[df['horizon'] == h]
        for metric in ['test_RMSE', 'test_MAE', 'test_IA']:
            vals = sub[metric].values
            print(f"  h={h:2d}h {metric:10s}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")

    print(f"\n保存: ./doc/multiseed_results.csv")


if __name__ == '__main__':
    main()
