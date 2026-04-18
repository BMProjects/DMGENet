# DMGENet

DMGENet is a paper-aligned research codebase for multi-station PM2.5 forecasting.

The repository has been intentionally reduced to the assets needed to reproduce
the manuscript-level experiments and to serve as a clean starting point for
follow-up research. Large intermediate branches, historical revision artifacts,
and non-paper result trees have been removed from the public-facing structure.

## Scope

This repository currently focuses on:

- unified dataset preparation for the paper datasets,
- four-graph construction (`D`, `N`, `S`, `F`),
- DMGENet base-model training,
- RL-guided graph-level integration,
- baseline training under the same evaluation protocol.

Supported paper datasets:

- `Beijing_12`
- `Beijing_Recent_12`
- `Chengdu_10`
- `Delhi_NCT_Meteo`

## Repository Layout

```text
.
├── baselines/          Baseline model implementations and the unified runner
├── data/               Dataset builders, compact-data utilities, and dataloaders
├── dataset/            Compact reproducible paper datasets
├── graphs/             Unified D/N/S/F graph protocol and graph builders
├── models/             DMGENet model components
├── pipelines/          Top-level experiment entry scripts
├── rlmc/               RL-guided integration data preparation and actor-critic code
├── utils/              Shared metrics and helper utilities
├── train_base.py       Base-model training entry point
├── train_rlmc.py       RL-guided integration entry point
└── setup_env.sh        Local environment bootstrap script
```

Additional project notes:

- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## Environment Setup

```bash
bash setup_env.sh
```

## Reproduction Status

The repository now follows a compact public-data layout:

- continuous station panels are kept under `dataset/*/AQI_processed/`,
- graph tensors and station metadata remain under `dataset/*/graphs/` and
  `dataset/*/location/`,
- large sliding-window tensors under `train_val_test_data/` are rebuilt
  locally from those compact station panels.

The compact station panels are the smallest self-contained dataset assets
needed to reproduce the paper-level training inputs without depending on the
author's private raw archives.

Each paper dataset also ships with a `split_manifest.json` file. These
manifests define the manuscript-aligned chronological train/validation/test
date ranges used when rebuilding `train_val_test_data/`.

## Paper Reproduction Workflow

### 1. Build compact datasets from external raw archives (optional)

This step is only needed when reconstructing the compact station panels from
the original external raw archives. A normal GitHub clone does not need it.

```bash
.venv/bin/python data/build_cn_dataset.py preprocess --city beijing_recent
.venv/bin/python data/build_cn_dataset.py preprocess --city chengdu
.venv/bin/python data/build_delhi_dataset.py --output-dataset Delhi_NCT_Meteo
```

### 2. Generate local sliding-window splits

The public repository keeps only the compact station panels. Before training,
generate the local NPZ splits:

```bash
.venv/bin/python data/compact_dataset.py --dataset all
```

This writes the local `train_val_test_data/72_{horizon}/...` directories for
`horizon in {1, 6, 12, 24}`. The same preparation step is also triggered
automatically by the dataloaders if those files are missing.

### 3. Build the unified graphs

```bash
.venv/bin/python graphs/build_graphs.py --dataset Beijing_12
.venv/bin/python graphs/build_graphs.py --dataset Beijing_Recent_12
.venv/bin/python graphs/build_graphs.py --dataset Chengdu_10
.venv/bin/python graphs/build_graphs.py --dataset Delhi_NCT_Meteo
```

### 4. Train the graph-specific DMGENet base predictors

```bash
.venv/bin/python train_base.py --dataset Beijing_12 --pred-lens 1 6 12 24
```

### 5. Prepare RL integration inputs

```bash
.venv/bin/python rlmc/prepare_data.py --dataset Beijing_12
.venv/bin/python rlmc/errors.py --dataset Beijing_12
```

### 6. Train graph-level integration

```bash
.venv/bin/python train_rlmc.py --dataset Beijing_12 --pred-lens 1 6 12 24
```

### 7. Run baselines

```bash
.venv/bin/python baselines/run_baseline.py --model agcrn --dataset Beijing_12 --pred-lens 1 6 12 24
```

## One-Command Pipeline

```bash
bash pipelines/run_unified_pipeline.sh
```

See [pipelines/README.md](pipelines/README.md) for the orchestration details.

## Project Philosophy

This repository should be read as a reproducible benchmark and engineering
artifact, not as a claim that the current method is the final solution to the
problem. The code is organized so that future work can reuse:

- the data pipeline,
- the graph protocol,
- the baseline adapters,
- the evaluation routines,
- and selected model components.

## Release Note

The repository is intended to stay lightweight in Git:

- compact station panels are tracked,
- large derived tensors are regenerated locally,
- and the train/eval entrypoints can rebuild those tensors automatically.
