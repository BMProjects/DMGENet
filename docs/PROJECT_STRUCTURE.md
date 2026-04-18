# Project Structure

This document summarizes the repository layout and the role of each module.

## Design Goal

The project is organized around a simple experimental workflow:

- dataset preparation,
- graph construction,
- DMGENet base-model training,
- graph-level integration,
- baseline evaluation,
- end-to-end orchestration.

## Main Directories

### `data/`
Dataset builders and dataloaders.

- `build_cn_dataset.py` builds the Chinese-city datasets.
- `build_delhi_dataset.py` builds the meteorology-enhanced Delhi dataset.
- `compact_dataset.py` rebuilds local `train_val_test_data/` NPZ splits.
- `dataloader.py` is the unified loader used by the current DMGENet training path.

### `graphs/`
Graph construction code.

- `unified_graphs.py` is the single source of truth for graph hyperparameters.
- `build_graphs.py` builds the `D`, `N`, `S`, and `F` graphs.
- `build_poi_graph.py` contains the OpenStreetMap-based functional-graph builder and validation utilities.

### `models/`
Current DMGENet model implementation.

- `dmgenet.py` defines the DMGENet architecture.
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

## Main Principle

Each top-level directory corresponds to a distinct part of the experimental
workflow. New files should follow that same logic so that collaborators can
identify their purpose from the path alone.
