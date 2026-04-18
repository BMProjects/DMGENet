"""
DMGENet base-model training entry point.

Trains four single-graph variants (Model_D, Model_N, Model_S, Model_POI) on
the PM2.5 forecasting benchmark. Graph matrices are loaded from the unified
`dataset/{dataset}/graphs/` directory produced by graphs/build_graphs.py.

Supported datasets: Beijing_12, Beijing_Recent_12, Chengdu_10, Delhi_NCT_Meteo.

Example:
  python train_base.py --dataset Beijing_12 --pred-lens 1 6 12 24
  python train_base.py --dataset Chengdu_10 --models Model_POI --ablation wo_ASTAM
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from models.dmgenet import Model
from graphs.unified_graphs import load_graph_tensors, resolve_graph_dir
from data.dataloader import CityDataLoader
from utils.metrics import metric_multi_sites
from utils.tools import adjust_learning_rate, EarlyStopping, setup_seed

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Graph loading
# ──────────────────────────────────────────────────────────────────────────────

_GRAPHS_CACHE: dict = {}


def load_city_graphs(dataset: str, graph_dir: str = None) -> dict:
    """Load precomputed D/N/S/POI graph tensors; cache to avoid repeated I/O."""
    cache_key = (dataset, str(resolve_graph_dir(ROOT, dataset, graph_dir)))
    if cache_key in _GRAPHS_CACHE:
        return _GRAPHS_CACHE[cache_key]

    graph_path = resolve_graph_dir(ROOT, dataset, graph_dir)
    if not graph_path.exists():
        print(f"ERROR: graph directory not found at {graph_path}")
        print("Run first: python graphs/build_graphs.py --dataset", dataset)
        sys.exit(1)

    graphs = load_graph_tensors(ROOT, dataset, DEVICE, graph_dir=graph_dir)
    print(f"Loaded graphs for {dataset}:")
    for name, t in graphs.items():
        n_edges = int((t > 0).sum().item())
        print(f"  {name}: shape={tuple(t.shape)}, edges={n_edges}")

    _GRAPHS_CACHE[cache_key] = graphs
    return graphs


# ──────────────────────────────────────────────────────────────────────────────
# Experiment driver
# ──────────────────────────────────────────────────────────────────────────────

class BaseModelExperiment:
    """Train + evaluate a single-graph DMGENet variant on one (dataset, horizon)."""

    def __init__(
        self,
        model_name:  str,
        model:       nn.Module,
        epoch:       int,
        learning_rate: float,
        target:      str,
        batch_size:  int,
        num_workers: int,
        seq_len:     int,
        predict_len: int,
        dataset:     str,
        results_folder_override: str = None,
        patience: int = 7,
    ):
        self.model_name    = model_name
        self.model         = model.to(DEVICE)
        self.epoch         = epoch
        self.learning_rate = learning_rate
        self.target        = target
        self.dataset       = dataset
        self.patience      = patience

        print(f"dataset:{dataset}  model:{model_name}  target:{target}  predict_len:{predict_len}")

        if results_folder_override is not None:
            self.results_folder = results_folder_override
        else:
            self.results_folder = os.path.join(
                f"./results/base_models/{dataset}", str(seq_len), str(predict_len), model_name
            )
        os.makedirs(self.results_folder, exist_ok=True)

        root_path = f"./dataset/{dataset}/train_val_test_data/{seq_len}_{predict_len}"
        self.train_dataloader = CityDataLoader(
            os.path.join(root_path, f"train_{target}.npz"), "train", batch_size, num_workers, target)
        self.val_dataloader   = CityDataLoader(
            os.path.join(root_path, f"val_{target}.npz"),   "val",   batch_size, num_workers, target)
        self.test_dataloader  = CityDataLoader(
            os.path.join(root_path, f"test_{target}.npz"),  "test",  batch_size, num_workers, target)

        self.train_loader = self.train_dataloader.get_dataloader()
        self.val_loader   = self.val_dataloader.get_dataloader()
        self.test_loader  = self.test_dataloader.get_dataloader()

    # ── val helper ─────────────────────────────────────────────────

    def val(self, criterion):
        val_loss = []
        self.model.eval()
        with torch.no_grad():
            for features, target in self.val_loader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                val_loss.append(criterion(self.model(features), target).item())
        self.model.train()
        return np.average(val_loss)

    # ── train ───────────────────────────────────────────────────────

    def train(self):
        criterion = nn.MSELoss()
        optim     = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses, val_losses, epoch_times = [], [], []
        self.model.train()
        t0 = time.time()
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True,
            path=os.path.join(self.results_folder, self.model_name + ".pth"),
        )

        for epoch in range(self.epoch):
            epoch_train_loss = []
            t_ep = time.time()
            for features, target in self.train_loader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                optim.zero_grad()
                loss = criterion(self.model(features), target)
                epoch_train_loss.append(loss.item())
                loss.backward()
                optim.step()

            ep_time = time.time() - t_ep
            epoch_times.append(ep_time)

            ep_train = np.average(epoch_train_loss)
            ep_val = self.val(criterion)
            train_losses.append(ep_train)
            val_losses.append(ep_val)

            print(
                f"Epoch [{epoch+1:3d}/{self.epoch}]  time:{ep_time:.1f}s  "
                f"train:{ep_train:.5f}  val:{ep_val:.5f}",
                end=" ",
            )
            adjust_learning_rate(optim, epoch + 1, self.learning_rate)
            early_stopping(ep_val, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        elapsed = time.time() - t0
        print(f"Training finished in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        pd.DataFrame({
            "epoch_time": epoch_times,
            "train_loss": train_losses,
            "val_loss":   val_losses,
        }).to_csv(os.path.join(self.results_folder, "loss.csv"), index=True, index_label="epoch")

        best_path = os.path.join(self.results_folder, self.model_name + ".pth")
        self.model.load_state_dict(torch.load(best_path))

    # ── evaluate ────────────────────────────────────────────────────

    def _evaluate(self, dataloader, loader, flag: str):
        predictions, trues, features_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for features, target in loader:
                features_list.append(features.numpy())
                pred = self.model(features.to(DEVICE))
                predictions.append(pred.cpu().numpy())
                trues.append(target.numpy())

        all_features = np.concatenate(features_list, axis=0)
        trues        = np.concatenate(trues,         axis=0)
        predictions  = np.concatenate(predictions,   axis=0)

        print(f"{flag}: features{all_features.shape}  pred{predictions.shape}  true{trues.shape}")

        np.save(os.path.join(self.results_folder, f"{flag}_X.npy"),           all_features)
        np.save(os.path.join(self.results_folder, f"{flag}_y.npy"),           trues)
        np.save(os.path.join(self.results_folder, f"{flag}_predictions.npy"), predictions)

        trues_inv = dataloader.inverse_transform(trues)
        preds_inv = dataloader.inverse_transform(predictions)

        np.save(os.path.join(self.results_folder, f"{flag}_y_inverse.npy"),           trues_inv)
        np.save(os.path.join(self.results_folder, f"{flag}_predictions_inverse.npy"), preds_inv)

        metrics = metric_multi_sites(preds_inv, trues_inv)
        print(f"{flag}  MAE:{metrics[0]:.3f}  RMSE:{metrics[1]:.3f}  IA:{metrics[2]:.4f}  R2:{metrics[3]:.3f}")

        pd.DataFrame([metrics], columns=[f"{flag}_MAE", f"{flag}_RMSE", f"{flag}_IA", f"{flag}_R2"]) \
          .to_csv(os.path.join(self.results_folder, f"{flag}_metrics.csv"), index=False)

    def test(self):
        self.model.eval()
        self._evaluate(self.val_dataloader,  self.val_loader,  "val")
        self._evaluate(self.test_dataloader, self.test_loader, "test")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DMGENet base-model training")
    parser.add_argument("--dataset",   required=True,
                        help="Dataset name, e.g. Beijing_12, Chengdu_10, Delhi_NCT_Meteo")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Number of nodes (optional; inferred from the graph matrix)")
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[1, 6, 12, 24],
                        help="Forecast horizons (default: 1 6 12 24)")
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--num-workers",type=int,   default=3)
    parser.add_argument("--seq-len",    type=int,   default=72)
    parser.add_argument("--seed",       type=int,   default=2026)
    parser.add_argument("--hidden-size",type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--num-heads",  type=int,   default=4)
    parser.add_argument("--apt-size",   type=int,   default=10)
    parser.add_argument("--num-blocks", type=int,   default=2)
    parser.add_argument("--results-dir", default=None,
                        help="Results root (default: ./results/base_models/{dataset})")
    parser.add_argument("--patience", type=int, default=7,
                        help="EarlyStopping patience (default: 7)")
    parser.add_argument("--graph-dir", default=None,
                        help="Unified graphs directory (default: dataset/{dataset}/graphs)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to specific variants, e.g. --models Model_POI")
    parser.add_argument("--ablation", default=None,
                        choices=["wo_T", "wo_GCN", "wo_GAT", "wo_ASTAM", "full"],
                        help=(
                            "Ablation variant: "
                            "wo_T = remove Gated-TCN (fall back to linear), "
                            "wo_GCN = remove GCN branch, "
                            "wo_GAT = remove GAT branch, "
                            "wo_ASTAM = remove adaptive spatio-temporal attention, "
                            "full = all components (equivalent to omitting this flag)"
                        ))
    return parser.parse_args()


# Ablation variant → (gated_TCN, gcn, gat, ASTAM) boolean flags.
ABLATION_FLAGS = {
    None:         (True,  True,  True,  True),
    "full":       (True,  True,  True,  True),
    "wo_T":       (False, True,  True,  True),
    "wo_GCN":     (True,  False, True,  True),
    "wo_GAT":     (True,  True,  False, True),
    "wo_ASTAM":   (True,  True,  True,  False),
}


def main():
    args = parse_args()

    IN_CHANNELS  = 12
    NUM_CHANNELS = [64, 64, 64, 64]

    setup_seed(args.seed)
    graphs = load_city_graphs(args.dataset, graph_dir=args.graph_dir)

    if args.num_nodes is None:
        sample_adj = next(iter(graphs.values()))
        args.num_nodes = sample_adj.shape[0]
        print(f"Inferred num_nodes={args.num_nodes}")

    abl = args.ablation
    gated_TCN, gcn, gat, astam = ABLATION_FLAGS[abl]
    abl_tag = abl if abl and abl != "full" else "full"
    print(f"Ablation: {abl_tag}  (gated_TCN={gated_TCN}, gcn={gcn}, gat={gat}, ASTAM={astam})")

    selected_models = args.models or list(graphs.keys())
    missing = [m for m in selected_models if m not in graphs]
    if missing:
        raise ValueError(f"Unknown models: {missing}. Available: {list(graphs.keys())}")

    for T_out in args.pred_lens:
        for model_name in selected_models:
            adj = graphs[model_name]
            model = Model(
                adj,
                input_size   = IN_CHANNELS,
                hidden_size  = args.hidden_size,
                dropout      = args.dropout,
                alpha        = 0.2,
                n_heads      = args.num_heads,
                num_channels = NUM_CHANNELS,
                apt_size     = args.apt_size,
                num_nodes    = args.num_nodes,
                num_block    = args.num_blocks,
                T_in         = args.seq_len,
                predict_len  = T_out,
                gated_TCN_bool = gated_TCN,
                gcn_bool       = gcn,
                gat_bool       = gat,
                ASTAM_bool     = astam,
            )

            # Route ablation outputs to a separate directory to avoid polluting main results.
            results_folder_override = None
            if args.results_dir:
                results_folder_override = os.path.join(
                    args.results_dir, str(args.seq_len), str(T_out), model_name
                )
            elif abl and abl != "full":
                results_folder_override = os.path.join(
                    "results", "ablation", args.dataset, abl_tag,
                    str(args.seq_len), str(T_out), model_name
                )

            exp = BaseModelExperiment(
                model_name   = model_name,
                model        = model,
                epoch        = args.epochs,
                learning_rate= args.lr,
                target       = "PM25",
                batch_size   = args.batch_size,
                num_workers  = args.num_workers,
                seq_len      = args.seq_len,
                predict_len  = T_out,
                dataset      = args.dataset,
                results_folder_override = results_folder_override,
                patience     = args.patience,
            )

            print(f"\n{'='*80}")
            print(f"{model_name}  pred_len={T_out}  START")
            print(f"{'='*80}")
            exp.train()
            exp.test()
            print(f"{model_name}  pred_len={T_out}  DONE")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
