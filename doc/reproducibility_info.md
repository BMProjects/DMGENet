# DMGENet 可复现性信息 (Section 5: Experimental Setup)

> 用于论文修订，回应 R1 (Comment 9), R2 (Comments 7, 13), R3

---

## Hardware & Software Environment

| Item | Specification |
|------|---------------|
| GPU  | NVIDIA GeForce RTX 4090 D (24 GB GDDR6X) |
| CPU  | (AMD/Intel, to be filled) |
| RAM  | (to be filled, e.g. 64 GB DDR5) |
| OS   | Ubuntu 22.04 LTS |
| Python | 3.10.19 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| NumPy | (to be filled) |
| Pandas | (to be filled) |

---

## Training Protocol

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate (initial) | 0.001 |
| LR schedule | Fixed for first 5 epochs; then × 0.8 per epoch |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 7 (based on validation loss) |
| Random seed (base models) | 2026 (single run), [42,123,456,789,2024] (multi-seed) |

## Base Model Architecture

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Input window (T_in) | 72 h | 3 days history |
| Input channels | 12 | meteorological + pollutant features |
| Hidden size | 64 | |
| TCN kernel size | 2 | (NOT 4 — corrected in paper Table 4) |
| TCN dilation rates | [1, 2, 4, 8] | 4 dilated conv layers |
| TCN output channels | [64, 64, 64, 64] | |
| GAT heads | 4 | concatenation + projection |
| GAT attention dropout | 0.2 | |
| GCN layers | 2 | conv1→ReLU→conv2 |
| Spatiotemporal blocks | 2 | |
| Node embedding size | 10 | adaptive graph |
| Total parameters | ~375K | varies by predict_len |

## Data Normalization

Min-max normalization per feature, fit on training split:
```
x_norm = (x - x_min) / (x_max - x_min)
```
Inverse transform applied before metric computation.

## Dataset Splits

| Split | Ratio | Beijing_12 | Delhi_NCT |
|-------|-------|------------|-----------|
| Train | ~60%  | 2013-03 to 2016-06 | 2020-01 to 2022-03 |
| Val   | ~20%  | 2016-07 to 2016-12 | 2022-04 to 2022-10 |
| Test  | ~20%  | 2017-01 to 2017-02 | 2022-11 to 2023-06 |

## RLMC Hyperparameters

| Parameter | Value |
|-----------|-------|
| DDPG state dimension | 12 (one per monitoring station) |
| DDPG action dimension | 4 (one weight per graph model) |
| Actor hidden dim | 64 |
| Critic hidden dim | 64 |
| γ (discount factor) | 0.99 |
| τ (soft update rate) | 0.005 |
| Actor learning rate | 1×10⁻⁴ |
| Critic learning rate | 1×10⁻³ |
| Replay buffer size | 2000 |
| Batch size | 64 |
| Training episodes | 200 |
| Max steps per episode | 1000 |
| Number of runs | 10 (best selected by val_loss) |
| Random seeds per run | 0–9 |

## Training Time (per horizon, RTX 4090 D)

| Stage | h=1 | h=6 | h=12 | h=24 |
|-------|-----|-----|------|------|
| Base model per graph | 2.8 min | 1.8 min | 1.7 min | 1.2 min |
| 4 base models total | 11.2 min | 7.2 min | 6.8 min | 4.8 min |
| RLMC (10 runs × 200 ep) | 59 min | 59 min | 59 min | 60 min |
| **Total per horizon** | **~70 min** | **~66 min** | **~66 min** | **~65 min** |
| **Full pipeline (4 horizons)** | — | — | — | **~267 min (~4.5h)** |

Average epochs before early stopping: 22.5 (h=1), 14.8 (h=6), 13.5 (h=12), 10.0 (h=24)

---

## Inference Time

From `doc/compute_efficiency.csv` (batch_size=64, RTX 4090 D):

| Model config | Single sample (ms) | Batch of 64 (ms) |
|---|---|---|
| Full model (h=1) | 4.19 | 5.73 |
| Full model (h=6) | 4.20 | 5.70 |
| Full model (h=12) | 4.21 | 5.72 |
| Full model (h=24) | 4.32 | 5.72 |

Total parameters: ~375K (trainable).

---

## Graph Construction Thresholds

| Graph | Threshold | Notes |
|-------|-----------|-------|
| Distance (D) | σ=16 km, ε=0.4 | Gaussian kernel, min-max norm |
| Neighbor (N) | radius=45 km | Binary, KNN-style |
| Similarity (S) | JS divergence threshold=0.6 | Distribution similarity |
| Functional (F/POI) | threshold=0.7 | POI category co-occurrence |
