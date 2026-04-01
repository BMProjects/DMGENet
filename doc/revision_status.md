# DMGENet Revision Status — April 1, 2026

> Deadline: April 16, 2026 · Days remaining: 15
> Updated: 2026-04-01 17:00 CST

---

## Code Fixes (P0) — ALL COMPLETE ✅

| Fix | File | Status |
|-----|------|--------|
| `F.tanh`/`F.sigmoid` deprecated → `torch.tanh`/`torch.sigmoid` | model/model_1.py:45,47 | ✅ Done |
| Double `.cuda()` on `node_embedding` | model/model_1.py:101 | ✅ Done |
| `.squeeze()` → `.squeeze(-1).squeeze(-1)` | model/model_1.py:120,124 | ✅ Done |
| `val()` missing `torch.no_grad()` | exp_base_model.py:83 | ✅ Done |
| Replay buffer stores `None` on `done=True` | RLMC_final/RLMC.py:134 | ✅ Done |
| Episode-level model selection: `test_loss` → `val_loss` | train_RLMC_final.py:140-174 | ✅ Done |
| Run-level model selection: `test MAE` → `val_loss` | train_RLMC_final.py:515-641 | ✅ Done |
| Add `predict_len=1` to training loop | train_RLMC_final.py:466 | ✅ Done |
| `plot_loss` non-interactive backend | utils/tools.py | ✅ Done |

---

## Experimental Results — ALL COMPLETE ✅

### Beijing 12-Station (Primary Dataset)

| Horizon | DMGENet RMSE | DMGENet MAE | DMGENet IA |
|---------|-------------|-------------|------------|
| 1h  | 16.30 (best run) / 16.36±0.06 (10 runs) | 8.74 / 8.82 | 0.990 |
| 6h  | 35.79 / 35.80±0.24 | 20.74 / 20.53 | 0.948 |
| 12h | 48.29 / 48.11±0.36 | 28.40 / 28.30 | 0.895 |
| 24h | 61.47 / 61.42±0.35 | 38.67 / 38.96 | 0.766 |

*Paper reported: 1h RMSE=16.48, 6h RMSE=45.41 (pre-fix with test-set leakage)*

### Delhi NCT (Generalization Dataset) — NEW ✅

| Horizon | DMGENet RMSE | DMGENet MAE | DMGENet IA |
|---------|-------------|-------------|------------|
| 1h  | 21.97 | 13.12 | 0.982 |
| 6h  | 38.47 | 23.68 | 0.939 |
| 12h | 46.38 | 28.81 | 0.907 |
| 24h | 52.08 | 32.88 | 0.873 |

*Note: Delhi outperforms Beijing at h=12 (46.38 vs 48.11) and h=24 (52.08 vs 61.42)*

### Multi-seed Base Models (5 seeds, in progress → ~5:20 PM)

Status: 70/80 models done (seed 2024 at h=6)
Partial results (4 seeds complete, very stable):
- h=1 RMSE CV: 0.77–1.73% (extremely stable)
- h=24 RMSE CV: 3.20–3.57% (acceptable)

### Statistical Significance

| Test | Result |
|------|--------|
| RLMC 10-run CV (h=1 RMSE) | 0.37% |
| RLMC 10-run CV (h=6 RMSE) | 0.67% |
| RLMC vs best single-graph, h=1 (Wilcoxon) | p=0.001 ✅ significant |
| RLMC vs best single-graph, h=6 (Wilcoxon) | p=0.188 ✗ not significant |
| RLMC vs best single-graph, h=12 (Wilcoxon) | p=0.999 ✗ not significant |
| RLMC vs best single-graph, h=24 (Wilcoxon) | p=0.002 ✅ significant |

---

## Analysis Outputs — ALL COMPLETE ✅

| Analysis | File | Purpose |
|----------|------|---------|
| Statistical significance | doc/statistical_significance.csv | R2-C10 |
| RLMC vs single-graph test | doc/rlmc_vs_single_test.csv | R2-C10 |
| Compute efficiency | doc/compute_efficiency.csv | R2-C8 |
| Noise robustness | doc/noise_robustness.csv | R2-C12 |
| Stationarity (ADF) | doc/stationarity_analysis.csv | R1-C5 |
| Station difficulty | doc/station_difficulty.csv | Discussion |
| Cross-region comparison | doc/cross_region_comparison.csv | R1-C1 |
| Multi-seed summary | doc/multiseed_results.csv | R2-C10 (in progress) |

---

## Paper Writing Tasks — PENDING

### P0 Paper Changes (must do before submission)

- [ ] **Eq.22**: Change normalization formula z-score → min-max
- [ ] **Table 4**: Update window=24 → 72, kernel_size=4 → 2, GCN layers=1 → 2
- [ ] **Eq.13**: Change GAT aggregation description: average → concat+projection
- [ ] **Figures**: Unify D/N/S/C → D/N/S/F throughout
- [ ] **ASTAM**: Simplify Eq.17-20 to match actual implementation
- [ ] **Title**: Remove all acronyms (e.g., "DMGENet" → spell out or rephrase)
- [ ] **Keywords**: Remove all acronyms; add "artificial intelligence" and "air quality forecasting"
- [ ] **Abstract**: Define all acronyms on first use (GNN, TCN, ASTAM, DDPG, MAE, RMSE, etc.)
- [ ] **Tables**: Update all result tables with new 4-horizon values + mean±std format

### P1 Paper Additions

- [ ] **Section 5.1**: Add Delhi NCT dataset description
- [ ] **Table X**: Cross-region generalization results
- [ ] **Table Y**: Multi-seed base model mean±std
- [ ] **Table Z**: Noise robustness (from doc/noise_robustness.csv)
- [ ] **Table W**: Compute efficiency (from doc/compute_efficiency.csv)
- [ ] **Section 5.2**: Expand with hardware specs + training protocol (from doc/reproducibility_info.md)
- [ ] **Section 5.4**: Expand Discussion with weight analysis + insights (from doc/discussion_draft.md)
- [ ] **Section 6 (new)**: Add Limitations section (from doc/discussion_draft.md)
- [ ] **Section 2**: Add ~16 new literature citations
- [ ] **Section 7**: Rewrite Conclusion with specific numbers

### D7 — SOTA Baselines (April 2, 2026)

Status: NOT STARTED — requires web search for public implementations

Priority baseline implementations to check:
1. GC-LSTM (already in paper) — no change needed
2. MSTGAN (already in paper) — no change needed
3. Graph Multi-Attention Network with Bayesian HPO (R3-C6 suggestion, Delhi)
4. STGNN-TCN (R3-C6 suggestion)
5. CNN-LSTM hybrid (R1-C3, R2-C13)

---

## Ready-to-Use Paper Content (from doc/)

1. **doc/rebuttal_skeleton.md**: Complete point-by-point response template
2. **doc/discussion_draft.md**: Section 5.4 discussion text
3. **doc/reproducibility_info.md**: Section 5.2 content
4. **doc/noise_robustness_summary.md**: Section 5.X content
5. **doc/cross_region_comparison.csv**: Table X data

---

## Git Log (last 5 commits)

```
caf34bf  Add noise robustness summary
61b2459  Update discussion draft with Delhi RLMC weights
5d55115  Add discussion draft with weight analysis and limitations
1a5416b  Add rebuttal skeleton, reproducibility info, and multi-seed summary
22e7ab5  Fix P0 code issues + add Delhi NCT generalization experiment
```
