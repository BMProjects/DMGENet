# DMGENet

DMGENet forecasts multi-station PM2.5 by modeling station relations with
four complementary graphs — distance (`D`), neighborhood (`N`), distribution
similarity (`S`), and functional similarity (`F`) — and dynamically combining
the four graph-wise predictions through a reinforcement-learning integrator
(RLMC).

## Datasets

Four datasets are supported:

| Dataset              | Stations | Period          |
| -------------------- | -------- | --------------- |
| `Beijing_12`         | 12       | 2013.03–2017.02 |
| `Beijing_Recent_12`  | 12       | 2020–2023       |
| `Chengdu_10`         | 8        | 2020–2023       |
| `Delhi_NCT_Meteo`    | 12       | 2017.09–2020.07 |

## Baselines

Four baselines are bundled and share the same benchmark protocol as DMGENet:

- `iTransformer`
- `STAEformer`
- `PM2.5-GNN`
- `MSTGAN`

## Benchmark Protocol

| Item            | Setting                     |
| --------------- | --------------------------- |
| Input length    | 72 hours                    |
| Forecast horizon| 1, 6, 12, 24 hours          |
| Target          | PM2.5                       |
| Metrics         | MAE, RMSE, IA               |
| Model selection | Validation-set early stopping |

## Repository Layout

```text
.
├── baselines/          Baseline implementations and the unified runner
├── data/               Dataset preparation utilities and dataloaders
├── dataset/            Dataset files, graph assets, and split manifests
├── graphs/             D / N / S / F graph construction
├── models/             DMGENet model components
├── pipelines/          End-to-end experiment scripts
├── rlmc/               Graph-integration inputs and the RL-guided integrator
├── utils/              Metrics and shared utilities
├── train_base.py       Train graph-specific DMGENet base predictors
├── train_rlmc.py       Train the graph-level integration module
└── setup_env.sh        Environment bootstrap script
```

## Installation

```bash
bash setup_env.sh
```

This creates a local `.venv/` with all required dependencies
(PyTorch, NumPy, pandas, scikit-learn, GeoPandas, NetworkX, and the Overpass
client used by the functional-graph builder). A CUDA-capable GPU is required
for baseline training; DMGENet itself can be run on CPU for small datasets.

## Quick Start

### 1. Prepare local dataset splits

```bash
.venv/bin/python data/compact_dataset.py --dataset all
```

### 2. Build graphs

```bash
.venv/bin/python graphs/build_graphs.py --dataset Beijing_12
.venv/bin/python graphs/build_graphs.py --dataset Beijing_Recent_12
.venv/bin/python graphs/build_graphs.py --dataset Chengdu_10
.venv/bin/python graphs/build_graphs.py --dataset Delhi_NCT_Meteo
```

### 3. Train DMGENet base predictors

```bash
.venv/bin/python train_base.py --dataset Beijing_12 --pred-lens 1 6 12 24
```

### 4. Prepare RL integration inputs

```bash
.venv/bin/python rlmc/prepare_data.py --dataset Beijing_12
.venv/bin/python rlmc/errors.py --dataset Beijing_12
```

### 5. Train graph-level integration

```bash
.venv/bin/python train_rlmc.py --dataset Beijing_12 --pred-lens 1 6 12 24
```

### 6. Run baselines

```bash
.venv/bin/python baselines/run_baseline.py --model itransformer --dataset Beijing_12 --horizon 1 6 12 24
.venv/bin/python baselines/run_baseline.py --model staeformer  --dataset Beijing_12 --horizon 1 6 12 24
.venv/bin/python baselines/run_baseline.py --model pm25_gnn    --dataset Beijing_12 --horizon 1 6 12 24
.venv/bin/python baselines/run_baseline.py --model mstgan      --dataset Beijing_12 --horizon 1 6 12 24
```

## One-Command Reproduction

```bash
bash pipelines/run_unified_pipeline.sh
```

See [pipelines/README.md](pipelines/README.md) for the pipeline details.

## Result Locations

| Component                         | Path                                                |
| --------------------------------- | --------------------------------------------------- |
| DMGENet graph-specific predictors | `results/base_models/{dataset}/{graph}/72/{horizon}/` |
| RL-guided graph integration       | `results/rlmc/{dataset}/proposed/72/{horizon}/`       |
| Baselines                         | `results/baselines/{model}/{dataset}/72/{horizon}/`   |

Each result directory contains the trained checkpoint, the training log,
validation/test metrics (`val_metrics.csv`, `test_metrics.csv`), inverse-
transformed predictions, and the full run configuration.

## Citation

If you use this repository, please cite the associated paper.

```text
[Citation information would be added after published.]
```
