# Project Structure

This document explains the current repository layout after the paper-alignment
cleanup.

## Design Goal

The repository is organized around reproducibility and clarity:

- one place for data preparation,
- one place for graph construction,
- one place for DMGENet training,
- one place for RL-guided integration,
- one place for baselines,
- one place for orchestration.

The current layout is clean and paper-aligned. The public-facing repository now
keeps compact continuous station panels plus graph assets under `dataset/`,
while the large `train_val_test_data/` tensors are regenerated locally from the
compact panels when needed.

## Main Directories

### `data/`
Dataset builders and dataloaders.

- `build_cn_dataset.py` builds the current Chinese-city datasets.
- `build_delhi_dataset.py` builds the meteorology-enhanced Delhi dataset.
- `compact_dataset.py` rebuilds local `train_val_test_data/` NPZ splits from
  the compact `AQI_processed/` station panels.
- `dataloader.py` is the unified loader used by the current DMGENet training path.

### `graphs/`
Graph construction code.

- `unified_graphs.py` is the single source of truth for graph hyperparameters.
- `build_graphs.py` builds the paper-aligned `D`, `N`, `S`, and `F` graphs.
- `build_poi_graph.py` contains the OpenStreetMap-based functional-graph builder and validation utilities.

### `models/`
Current DMGENet model implementation.

- `dmgenet.py` defines the paper-aligned architecture.
- `tcn.py`, `gcn.py`, `gat.py`, and `causal_cnn.py` contain reusable model blocks.

### `rlmc/`
RL-guided integration utilities.

- `prepare_data.py` stacks base-model predictions for integration training.
- `errors.py` computes per-model error histories.
- `actor_critic.py` implements the actor-critic components used by `train_rlmc.py`.

### `baselines/`
Baseline implementations and the unified baseline runner.

### `pipelines/`
Shell entrypoints for full experiment runs.

### `utils/`
Shared support code such as metrics, adjacency normalization, random seed
utilities, and early stopping helpers.

## Main Entry Points

### `train_base.py`
Train the four graph-specific DMGENet base predictors.

### `train_rlmc.py`
Train the graph-level integration module using precomputed base-model outputs.

### `baselines/run_baseline.py`
Train and evaluate the baseline family under the unified protocol.

## What Was Removed

The repository no longer treats manuscript drafting artifacts, revision-only
assets, and large intermediate experiment branches as part of the public-facing
core. Legacy directories that only contained stale `__pycache__` content have
also been removed.

## Guiding Principle for Future Changes

If a new file or directory is added, it should be easy to answer:

1. Is it needed to reproduce the paper?
2. Is it a reusable research asset for follow-up work?
3. Would a new collaborator understand why it exists from its location alone?

If the answer is "no" to all three, it probably does not belong in the root repository.
