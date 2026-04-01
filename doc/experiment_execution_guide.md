# DMGENet 实验验证执行指南

> 本文件是面向执行者的具体操作手册，与 `revision_master_plan.md` (策略层) 配合使用。

---

## 前置条件

### 1. 环境安装

```bash
# 建议使用 conda 环境
conda create -n dmgenet python=3.10 -y
conda activate dmgenet

# PyTorch (根据 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy pandas scipy scikit-learn matplotlib requests dtaidistance thop
```

### 2. GPU 要求
- 最低: 1 × GPU, 8GB VRAM (如 RTX 3060/4060)
- 推荐: 1 × GPU, 16GB+ VRAM (如 RTX 4080/A100)
- 预计训练时间:
  - 基础模型: ~10 min/model × 20 models × 4 horizons ≈ 13 小时
  - RLMC 集成: ~5 min/run × 10 runs × 4 horizons ≈ 3 小时
  - 消融实验: 已包含在基础模型中

---

## Phase 0: 数据准备

### Step 0.1 — 下载 UCI Beijing 数据

```bash
cd /home/bm/Dev/DMGENet
python data/download_beijing_data.py
```

**预期输出目录结构**:
```
dataset/Beijing_12/
├── AQI_processed/           ← 每站 CSV (供图构建)
│   ├── PRSA_Data_1.csv      (Aotizhongxin)
│   ├── PRSA_Data_2.csv      (Changping)
│   └── ... (共 12 个)
├── train_val_test_data/     ← 滑窗数据 (供模型训练)
│   ├── 72_1/
│   │   ├── train_PM25.npz
│   │   ├── val_PM25.npz
│   │   ├── test_PM25.npz
│   │   └── scaler_PM25.npy
│   ├── 72_6/
│   ├── 72_12/
│   └── 72_24/
├── location/location.csv     ← 已有
├── POI/adjacency_matrix.csv  ← 已有
└── neighbors/neighbors.csv   ← 已有
```

### Step 0.2 — 验证数据

```python
import numpy as np

# 检查形状
d = np.load("dataset/Beijing_12/train_val_test_data/72_6/train_PM25.npz")
print(f"X shape: {d['X'].shape}")  # 期望: (samples, 12, 72, 12) = (N_samples, stations, seq_len, features)
print(f"y shape: {d['y'].shape}")  # 期望: (samples, 12, 6) = (N_samples, stations, pred_len)

scaler = np.load("dataset/Beijing_12/train_val_test_data/72_6/scaler_PM25.npy")
print(f"scaler: {scaler}")  # 期望: [min, max]
```

### Step 0.3 — 验证图构建

```python
# 验证 4 种图可以正常构建
python -c "
from _Support.Graph_Construction_Beijing_12 import *
adj_D, _, _ = calculate_the_distance_matrix(threshold=0.4)
print(f'Distance graph: {adj_D.shape}, edges={int(adj_D.sum())}')
adj_N, _, _ = calculate_the_neighbor_matrix(45)
print(f'Neighbor graph: {adj_N.shape}, edges={int(adj_N.sum())}')
adj_S, _, _ = calculate_the_similarity_matrix(threshold=0.6, target='PM25')
print(f'Similarity graph: {adj_S.shape}, edges={int(adj_S.sum())}')
import pandas as pd, torch
adj_F = pd.read_csv('./dataset/Beijing_12/POI/adjacency_matrix.csv', header=None)
adj_F = torch.where(torch.tensor(adj_F.values)>=0.7, torch.tensor(adj_F.values), torch.zeros(12,12))
print(f'Functional graph: {adj_F.shape}, edges={int((adj_F>0).sum())}')
"
```

---

## Phase 1: 基础模型训练 (复现 Table 2 基础)

### Step 1.1 — 训练完整模型 (4 图 × 4 horizons)

```bash
# 这将训练 20 个模型变体 (4图 × (proposed + 4 ablation)) × 4 horizons
# 但首先只跑 proposed 版本验证
python exp_base_model.py
```

**⚠️ 注意**: `exp_base_model.py` 当前配置:
- `T_out in [1, 6, 12, 24]` — 4 个预测步长
- 包含 proposed + 16 个消融变体 = 20 个模型
- 每个模型 ~100 epoch (带 early stopping, patience=7)
- 预计总训练时间: **10-15 小时** (单 GPU)

**建议**: 首次验证时只跑 `[1, 6]` + proposed 模型, 确认结果合理后再跑全部:

```python
# 临时修改 exp_base_model.py:250
for T_out in [1, 6]:  # 先跑两个 horizon
    models = {
        'Model_D': ...,
        'Model_N': ...,
        'Model_S': ...,
        'Model_POI': ...,
        # 先注释掉消融模型
    }
```

### Step 1.2 — 检查基础模型结果

训练完成后检查:
```bash
# 查看每个模型的测试指标
find 预测结果_基础模型_Beijing_12/ -name "test_metrics.csv" | sort | xargs -I {} sh -c 'echo "--- {} ---" && cat {}'
```

**论文 Table 2 中的基线对比**:
| 模型 | 1h RMSE | 1h MAE | 6h RMSE | 6h MAE |
|------|---------|--------|---------|--------|
| GC-LSTM | 16.8227 | 8.9341 | 50.2023 | 28.8114 |
| STCN | 16.7573 | 8.9251 | 50.0023 | 28.8114 |
| MSTGAN | 16.8610 | 9.1708 | 51.0602 | 29.9692 |
| **DMGENet** | **16.4829** | **8.6627** | **45.4055** | **26.3972** |

---

## Phase 2: RLMC 集成训练 (复现最终 DMGENet 结果)

### Step 2.1 — 准备 RLMC 输入数据

```bash
# 合并 4 个图模型的预测结果
python RLMC_final/get_X_y.py

# 计算历史误差
python RLMC_final/calculating_errors.py
```

**检查输出**:
```bash
ls RLMC_final_数据集_Beijing_12/proposed/72/6/
# 期望: val_X.npy, val_y.npy, val_predictions_all.npy,
#       test_X.npy, test_y.npy, test_predictions_all.npy,
#       combined_val_mae_history_errors.csv,
#       combined_test_mae_history_errors.csv, ...
```

### Step 2.2 — 训练 RLMC

```bash
python train_RLMC_final.py
```

**注意**: 代码已修复测试集泄漏问题, 现在:
- Episode 级: 基于 **val_loss** (RL 训练数据上的 loss) 选择最佳 episode
- Run 级: 基于 **val_loss** 选择最佳 run
- 测试集仅用于最终指标报告

### Step 2.3 — 检查 RLMC 结果

```bash
# 查看最终结果
cat RLMC_final_预测结果_Beijing_12/proposed/72/6/best_metrics.csv
cat RLMC_final_预测结果_Beijing_12/proposed/72/6/summary_metrics.csv  # mean ± std
```

---

## Phase 3: 结果对比与论文数据核实

### Step 3.1 — 汇总

```bash
python run_experiments.py --phase summary --horizons 1 6 12 24
```

### Step 3.2 — 关键对照检查清单

| 检查项 | 论文位置 | 代码输出位置 | 状态 |
|--------|---------|-------------|------|
| DMGENet 1-6h 结果 | Table 2 | RLMC_final_预测结果/proposed/ | ⬜ |
| 单图模型对比 | Table 5 | 预测结果_基础模型/ 各 Model_X | ⬜ |
| 组件消融 | Table 6 | 预测结果_基础模型/ 各 wo_X | ⬜ |
| 集成方法对比 | Table 7 | RLMC_final_预测结果/ (需额外跑 GWO/PSO/GA) | ⬜ |
| 12 站预测曲线 | Fig 7-9 | best_pred.npy / best_true.npy | ⬜ |
| 消融柱状图 | Fig 10 | 各 wo_X 模型结果汇总 | ⬜ |

### Step 3.3 — 结果差异分析

如果复现结果与论文差异较大 (>5%), 可能的原因:
1. **随机种子不同**: 基础模型使用 `setup_seed(2026)`, RLMC 使用 seed 0-9
2. **数据预处理差异**: 缺失值处理、归一化范围
3. **测试集泄漏修复**: 修复后 RLMC 结果可能下降 1-3%
4. **特征维度**: 风向编码方式可能不同

---

## 后续实验计划 (审稿人要求)

### 已支持 (只需跑实验):
- [x] 12h 和 24h 预测 (`T_out in [12, 24]` 已在代码中)
- [x] 10 次重复运行 mean±std (`repeat_times=10`)
- [x] 集成方法对比: Average, GWO, PSO, GA vs RLMC (需取消 `train_RLMC_final.py` 注释)

### 需新增代码:
- [ ] 更强基线模型 (需新增模型代码)
- [ ] 计算效率分析 (参数量/FLOPs/推理时间)
- [ ] 噪声鲁棒性实验
- [ ] 超参数敏感性分析
- [ ] 新数据集验证 (如有条件)

### 纯论文修改 (不需代码):
- [ ] Related Works 扩展 (~16 篇新引用)
- [ ] Limitations 章节
- [ ] Discussion 深化
- [ ] 缩写格式修正
- [ ] 语言润色

---

## 快速启动命令 (copy-paste ready)

```bash
# ===== 最小验证流程 (约 2 小时) =====

# 1. 准备数据
python data/download_beijing_data.py

# 2. 验证图构建
python -c "from _Support.Graph_Construction_Beijing_12 import *; print('OK')"

# 3. 只跑 1 个 horizon 的 proposed 模型
# (手动修改 exp_base_model.py 中 T_out 列表为 [6])
python exp_base_model.py

# 4. 准备 RLMC 数据
python RLMC_final/get_X_y.py
python RLMC_final/calculating_errors.py

# 5. 跑 RLMC (仅 3 次重复快速验证)
# (手动修改 train_RLMC_final.py 中 repeat_times=3)
python train_RLMC_final.py

# 6. 查看结果
cat RLMC_final_预测结果_Beijing_12/proposed/72/6/best_metrics.csv
```

```bash
# ===== 完整复现流程 (约 16+ 小时) =====
python data/download_beijing_data.py
python exp_base_model.py                    # 所有模型 × 所有 horizon
python RLMC_final/get_X_y.py
python RLMC_final/calculating_errors.py
python train_RLMC_final.py                  # 10 runs × 所有 horizon
python run_experiments.py --phase summary --horizons 1 6 12 24
```
