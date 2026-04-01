"""
计算效率分析 — 参数量、FLOPs、训练/推理时间
回应 Reviewer #2 Comment 8: 计算成本分析
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from model.model_1 import Model
from _Support.Graph_Construction_Beijing_12 import (
    calculate_the_distance_matrix,
    calculate_the_neighbor_matrix,
    calculate_the_similarity_matrix,
)

# ─── 配置 ─────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_NODES = 12
IN_CHANNELS = 12
HIDDEN_SIZE = 64
NUM_CHANNELS = [64, 64, 64, 64]
DROPOUT = 0.2
ALPHA = 0.2
NUM_HEADS = 4
APT_SIZE = 10
NUM_BLOCK = 2
T_IN = 72
BATCH_SIZE = 64
WARMUP_RUNS = 20
MEASURE_RUNS = 100

# ─── 图矩阵 ──────────────────────────────────────────────
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

adj_map = {'Model_D': adj_D, 'Model_N': adj_N, 'Model_S': adj_S, 'Model_POI': adj_POI}

# ─── 模型配置 ─────────────────────────────────────────────
model_configs = {
    'Full Model': dict(gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
    'w/o Gated TCN': dict(gated_TCN_bool=False, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
    'w/o GCN':       dict(gated_TCN_bool=True, gcn_bool=False, gat_bool=True, ASTAM_bool=True),
    'w/o GAT':       dict(gated_TCN_bool=True, gcn_bool=True, gat_bool=False, ASTAM_bool=True),
    'w/o ASTAM':     dict(gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=False),
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_time(model, x, warmup=WARMUP_RUNS, runs=MEASURE_RUNS):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / runs


def measure_gpu_memory(model, x):
    if DEVICE != 'cuda':
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        model(x)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MiB
    return peak


def main():
    results = []

    for predict_len in [1, 6, 12, 24]:
        x = torch.randn(BATCH_SIZE, NUM_NODES, T_IN, IN_CHANNELS).to(DEVICE)
        x_single = torch.randn(1, NUM_NODES, T_IN, IN_CHANNELS).to(DEVICE)

        for config_name, config in model_configs.items():
            # 使用 Distance 图做代表（4种图的模型结构完全一致）
            model = Model(
                adj_D, IN_CHANNELS, HIDDEN_SIZE, DROPOUT, ALPHA, NUM_HEADS,
                num_channels=NUM_CHANNELS, apt_size=APT_SIZE,
                num_nodes=NUM_NODES, num_block=NUM_BLOCK,
                T_in=T_IN, predict_len=predict_len,
                **config
            ).to(DEVICE)

            total_params, trainable_params = count_params(model)
            infer_time_batch = measure_time(model, x) * 1000       # ms
            infer_time_single = measure_time(model, x_single) * 1000  # ms
            gpu_mem = measure_gpu_memory(model, x)

            row = {
                'horizon': predict_len,
                'model': config_name,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'infer_time_batch_ms': round(infer_time_batch, 2),
                'infer_time_single_ms': round(infer_time_single, 2),
                'gpu_peak_MiB': round(gpu_mem, 1),
                'batch_size': BATCH_SIZE,
            }
            results.append(row)
            print(f"  h={predict_len:2d}h  {config_name:15s}  params={trainable_params:>9,}  "
                  f"infer={infer_time_single:.2f}ms/sample  mem={gpu_mem:.0f}MiB")

            del model
            torch.cuda.empty_cache()

    # 额外：4 个独立图模型组合的总计（DMGENet 整体）
    total_single_params = results[0]['trainable_params']  # Full Model
    dmgenet_total = total_single_params * 4
    print(f"\n  DMGENet 总参数量 (4 graphs): {dmgenet_total:,}")
    print(f"  单图模型参数量:              {total_single_params:,}")

    # 保存
    df = pd.DataFrame(results)
    out_path = './doc/compute_efficiency.csv'
    df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

    # 打印论文用表格
    print("\n" + "=" * 80)
    print("论文 Table: Computational Efficiency Analysis")
    print("=" * 80)
    sub = df[df['horizon'] == 6]  # 以 6h 为代表
    print(f"{'Model':20s} {'Params':>12s} {'Infer(ms)':>12s} {'GPU(MiB)':>10s}")
    print("-" * 56)
    for _, r in sub.iterrows():
        print(f"{r['model']:20s} {r['trainable_params']:>12,} {r['infer_time_single_ms']:>12.2f} {r['gpu_peak_MiB']:>10.1f}")
    print(f"{'DMGENet (4 graphs)':20s} {dmgenet_total:>12,} {'~' + str(round(sub.iloc[0]['infer_time_single_ms'] * 4, 1)):>12s} {'~' + str(round(sub.iloc[0]['gpu_peak_MiB'] * 1.3)):>10s}")


if __name__ == '__main__':
    main()
