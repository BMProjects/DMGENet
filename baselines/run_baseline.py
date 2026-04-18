"""
baselines/run_baseline.py
==========================
Unified baseline training and evaluation script.

This version does not force every method into one simplified tensor format.
Instead, each baseline uses a local adapter that stays close to its original
input design:
  - iTransformer: PM2.5 multivariate series + calendar covariates
  - STAEformer:   PM2.5 + TOD + DOW
  - PM2.5-GNN:    PM2.5 history + future exogenous features + transport graph
  - MSTGAN:       multi-scale graph-temporal modeling

Supported baselines:
  - iTransformer / STAEformer / PM2.5-GNN / MSTGAN

Output directory:
  results/baselines/{model}/{dataset}/72/{horizon}/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.data_adapter import (
    get_dataloaders,
    get_dataset_info,
    inverse_transform,
    inverse_transform_zscore,
)
from utils.metrics import metric_multi_sites
from utils.tools import EarlyStopping, setup_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

BASE_PRESETS = {
    "itransformer": {
        "epochs": 20,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "batch_size": 32,
        "patience": 7,
        "criterion": "mse",
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 2,
        "d_ff": 2048,
        "dropout": 0.1,
        "use_norm": True,
    },
    "staeformer": {
        "epochs": 80,
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "batch_size": 16,
        "patience": 12,
        "criterion": "huber",
        "input_embedding_dim": 24,
        "tod_embedding_dim": 24,
        "dow_embedding_dim": 24,
        "adaptive_embedding_dim": 80,
        "feed_forward_dim": 256,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.1,
        "use_mixed_proj": True,
        "steps_per_day": 24,
    },
    "pm25_gnn": {
        "epochs": 45,
        "lr": 5e-4,
        "weight_decay": 5e-4,
        "batch_size": 32,
        "patience": 7,
        "criterion": "mse",
        "hidden_dim": 64,
        "gnn_out": 13,
    },
    "mstgan": {
        "epochs": 40,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "patience": 7,
        "criterion": "mae",
        "block1_hidden": 64,
        "block2_hidden": 64,
        "cheb_k": 3,
        "d_model": 512,
        "dropout": 0.1,
    },
}


def get_runtime_hardware() -> dict:
    info = {
        "device": DEVICE,
        "cpu_count": os.cpu_count() or 4,
        "gpu_name": None,
        "gpu_mem_gb": 0.0,
        "is_4090_class": False,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        name = (info["gpu_name"] or "").lower()
        info["is_4090_class"] = ("4090" in name) or (info["gpu_mem_gb"] >= 20.0)
    return info


def format_cuda_debug_info() -> str:
    return (
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )


def get_model_presets(model_name: str, hardware: dict) -> dict:
    preset = dict(BASE_PRESETS[model_name])
    if hardware["device"] == "cuda":
        if hardware["is_4090_class"]:
            tuned_batch = {
                "itransformer": 192,
                "staeformer": 48,
                "pm25_gnn": 256,
                "mstgan": 64,
            }
        else:
            tuned_batch = {
                "itransformer": 64,
                "staeformer": 32,
                "pm25_gnn": 64,
                "mstgan": 32,
            }
        preset["batch_size"] = tuned_batch[model_name]
    return preset


def default_loader_settings(model_name: str, hardware: dict) -> tuple[int, int]:
    cpu_count = os.cpu_count() or 4
    if hardware["device"] != "cuda":
        return 0, 2
    if hardware["is_4090_class"]:
        tuned = {
            "itransformer": (10, 4),
            "staeformer": (8, 2),
            "pm25_gnn": (8, 2),
            "mstgan": (6, 2),
        }
        return tuned[model_name]
    return max(4, min(8, cpu_count - 1)), 2


def move_tensor(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device, non_blocking=torch.cuda.is_available())


def build_optimizer(model_name: str, model: nn.Module, lr: float, weight_decay: float):
    use_fused = torch.cuda.is_available()
    kwargs = {"lr": lr, "weight_decay": weight_decay, "eps": 1e-8}
    if use_fused:
        kwargs["fused"] = True
    return torch.optim.Adam(model.parameters(), **kwargs)


def get_amp_dtype(args):
    if not torch.cuda.is_available() or not args.use_amp:
        return None
    if args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def build_model(model_name: str, info: dict, horizon: int, args) -> nn.Module:
    n_nodes = info["n_nodes"]
    seq_len = info["seq_len"]

    if model_name == "itransformer":
        from baselines.itransformer.model import iTransformer

        return iTransformer(
            n_variates=n_nodes,
            seq_len=seq_len,
            pred_len=horizon,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            use_norm=args.use_norm,
        )

    if model_name == "staeformer":
        from baselines.staeformer.model import STAEformer

        return STAEformer(
            num_nodes=n_nodes,
            in_steps=seq_len,
            out_steps=horizon,
            steps_per_day=args.steps_per_day,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=args.input_embedding_dim,
            tod_embedding_dim=args.tod_embedding_dim,
            dow_embedding_dim=args.dow_embedding_dim,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=args.adaptive_embedding_dim,
            feed_forward_dim=args.feed_forward_dim,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            dropout=args.dropout,
            use_mixed_proj=args.use_mixed_proj,
        )

    if model_name == "pm25_gnn":
        from baselines.pm25_gnn.model import PM25GNN

        wind_mean = None
        wind_std = None
        if getattr(args, "use_wind", False):
            wind_mean = args.wind_mean
            wind_std = args.wind_std

        return PM25GNN(
            hist_len=seq_len,
            pred_len=horizon,
            in_dim=1 + args.feature_dim,
            city_num=n_nodes,
            edge_index=args.edge_index,
            edge_attr=args.edge_attr,
            wind_mean=wind_mean,
            wind_std=wind_std,
            use_wind=args.use_wind,
            hid_dim=args.hidden_dim,
            gnn_out=args.gnn_out,
        )

    if model_name == "mstgan":
        from baselines.mstgan.model import MSTGAN

        return MSTGAN(
            input_dim=args.mstgan_input_dim,
            block1_hidden=args.block1_hidden,
            block2_hidden=args.block2_hidden,
            num_nodes=n_nodes,
            num_of_timesteps=seq_len,
            pred_len=horizon,
            K=args.cheb_k,
            dropout=args.dropout,
            d_model=args.d_model,
            output_dim=1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_criterion(name: str) -> nn.Module:
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unknown criterion: {name}")


def forward_model(model_name: str, model: nn.Module, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name == "itransformer":
        x_enc = move_tensor(batch["x_enc"], device)
        x_mark_enc = move_tensor(batch["x_mark_enc"], device)
        y = move_tensor(batch["y"], device)  # (B,N,H)
        pred = model(x_enc, x_mark_enc=x_mark_enc)
        return pred, y

    if model_name == "staeformer":
        x = move_tensor(batch["x"], device)
        y = move_tensor(batch["y"], device)              # (B,H,N,1)
        pred = model(x)                        # (B,N,H)
        target = y[..., 0].permute(0, 2, 1)    # (B,N,H)
        return pred, target

    if model_name == "pm25_gnn":
        pm25_hist = move_tensor(batch["pm25_hist"], device)
        feature = move_tensor(batch["feature"], device)
        y = move_tensor(batch["y"], device)              # (B,N,H)
        pred = model(pm25_hist, feature)
        return pred, y

    if model_name == "mstgan":
        x = move_tensor(batch["x"], device)              # (B,N,F,T)
        y = move_tensor(batch["y"], device)              # (B,N,H)
        pred = model(x, model._cheb_polys)               # (B,N,H)
        return pred, y

    raise ValueError(f"Unknown model: {model_name}")


def build_scheduler(model_name: str, optimizer: torch.optim.Optimizer, args=None):
    if model_name == "staeformer":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    return None


def train_one_horizon(model_name: str, dataset: str, horizon: int, args, out_dir: Path):
    train_loader = val_loader = test_loader = None
    info = get_dataset_info(dataset)
    try:
        train_loader, val_loader, test_loader, data_meta = get_dataloaders(
            model_name=model_name,
            dataset=dataset,
            horizon=horizon,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            loader_config={"prefetch_factor": args.prefetch_factor},
        )
        if data_meta.inverse_kind == "minmax":
            sc_min = data_meta.pm25_min
            sc_range = data_meta.pm25_range
            pm25_mean = pm25_std = None
        else:
            sc_min = sc_range = None
            pm25_mean = data_meta.pm25_mean
            pm25_std = data_meta.pm25_std

        args.feature_dim = data_meta.feature_dim
        args.edge_index = data_meta.edge_index
        args.edge_attr = data_meta.edge_attr
        args.use_wind = bool(data_meta.use_wind)
        args.mstgan_input_dim = data_meta.mstgan_input_dim
        if data_meta.use_wind:
            feature_mean_flat = np.asarray(data_meta.feature_mean).reshape(-1)
            feature_std_flat = np.asarray(data_meta.feature_std).reshape(-1)
            args.wind_mean = (
                float(feature_mean_flat[-2]),
                float(feature_mean_flat[-1]),
            )
            args.wind_std = (
                float(feature_std_flat[-2]),
                float(feature_std_flat[-1]),
            )
        else:
            args.wind_mean = None
            args.wind_std = None

        model = build_model(model_name, info, horizon, args).to(DEVICE)

        if model_name == "mstgan":
            model._cheb_polys = [
                torch.from_numpy(p).float().to(DEVICE) for p in data_meta.cheb_polynomials
            ]

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {n_params:,}")

        criterion = get_criterion(args.criterion)
        optimizer = build_optimizer(model_name, model, args.lr, args.weight_decay)
        scheduler = build_scheduler(model_name, optimizer, args)
        amp_dtype = get_amp_dtype(args)
        use_amp = amp_dtype is not None
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

        out_dir.mkdir(parents=True, exist_ok=True)
        best_path = out_dir / "best_model.pth"
        early_stop = EarlyStopping(patience=args.patience, verbose=True, path=str(best_path))

        log_rows = []
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = []
            t_ep = time.time()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(amp_dtype):
                    pred, target = forward_model(model_name, model, batch, DEVICE)
                    loss = criterion(pred, target)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                train_loss.append(loss.item())

            model.eval()
            val_loss = []
            with torch.no_grad():
                for batch in val_loader:
                    with autocast_context(amp_dtype):
                        pred, target = forward_model(model_name, model, batch, DEVICE)
                        val_loss.append(criterion(pred, target).item())

            ep_train = float(np.mean(train_loss))
            ep_val = float(np.mean(val_loss))
            ep_time = time.time() - t_ep
            if scheduler is not None:
                scheduler.step()

            row = {"epoch": epoch, "train_loss": ep_train, "val_loss": ep_val, "epoch_time": ep_time}
            print(f"  Epoch [{epoch:3d}/{args.epochs}]  {ep_time:.1f}s  train:{ep_train:.5f}  val:{ep_val:.5f}")
            log_rows.append(row)

            early_stop(ep_val, model)
            if early_stop.early_stop:
                print("  Early stopping.")
                break

        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()

        def evaluate(loader, split_name: str):
            preds, trues = [], []
            with torch.no_grad():
                for batch in loader:
                    with autocast_context(amp_dtype):
                        pred, target = forward_model(model_name, model, batch, DEVICE)
                    preds.append(pred.detach().float().cpu().numpy())
                    trues.append(target.detach().float().cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            if data_meta.inverse_kind == "minmax":
                preds_inv = inverse_transform(preds, sc_min, sc_range)
                trues_inv = inverse_transform(trues, sc_min, sc_range)
            else:
                preds_inv = inverse_transform_zscore(preds, pm25_mean, pm25_std)
                trues_inv = inverse_transform_zscore(trues, pm25_mean, pm25_std)

            mae, rmse, ia, r2 = metric_multi_sites(preds_inv, trues_inv)
            print(f"  {split_name:5s}: MAE={mae:.4f}  RMSE={rmse:.4f}  IA={ia:.4f}  R2={r2:.4f}")

            pd.DataFrame(
                [{f"{split_name}_MAE": mae, f"{split_name}_RMSE": rmse, f"{split_name}_IA": ia, f"{split_name}_R2": r2}]
            ).to_csv(out_dir / f"{split_name}_metrics.csv", index=False)

            if split_name == "test":
                np.save(out_dir / "test_predictions_inverse.npy", preds_inv)
                np.save(out_dir / "test_y_inverse.npy", trues_inv)

        evaluate(val_loader, "val")
        evaluate(test_loader, "test")

        config = {
            "model": model_name,
            "dataset": dataset,
            "horizon": horizon,
            "seq_len": 72,
            "n_params": n_params,
            "epochs_trained": len(log_rows),
        }
        config.update(vars(args))
        config["runtime_device"] = DEVICE
        config["use_amp_effective"] = use_amp
        config["amp_dtype_effective"] = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    finally:
        for loader in (train_loader, val_loader, test_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
                loader._iterator = None


def parse_args():
    p = argparse.ArgumentParser(description="Run an official-style adapted baseline for PM2.5 forecasting")
    p.add_argument(
        "--model",
        required=True,
        choices=["itransformer", "staeformer", "pm25_gnn", "mstgan"],
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", nargs="+", type=int, default=[1, 6, 12, 24])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--use-amp", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None)
    p.add_argument("--allow-cpu", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)

    # training (None means use official preset)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)

    # shared model args
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--n-heads", type=int, default=None)

    # iTransformer
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--use-norm", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)

    # STAEformer
    p.add_argument("--input-embedding-dim", type=int, default=None)
    p.add_argument("--tod-embedding-dim", type=int, default=None)
    p.add_argument("--dow-embedding-dim", type=int, default=None)
    p.add_argument("--adaptive-embedding-dim", type=int, default=None)
    p.add_argument("--feed-forward-dim", type=int, default=None)
    p.add_argument("--steps-per-day", type=int, default=None)
    p.add_argument("--use-mixed-proj", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)

    # MSTGAN
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--cheb-k", type=int, default=None)
    p.add_argument("--gnn-out", type=int, default=None)
    p.add_argument("--block1-hidden", type=int, default=None)
    p.add_argument("--block2-hidden", type=int, default=None)

    args = p.parse_args()
    hardware = get_runtime_hardware()
    preset = get_model_presets(args.model, hardware)
    for key, value in preset.items():
        arg_key = key.replace("-", "_")
        if getattr(args, arg_key, None) is None:
            setattr(args, arg_key, value)
    args.criterion = preset["criterion"]
    default_workers, default_prefetch = default_loader_settings(args.model, hardware)
    if args.num_workers is None:
        args.num_workers = default_workers
    if args.prefetch_factor is None:
        args.prefetch_factor = default_prefetch
    if args.use_amp is None:
        args.use_amp = torch.cuda.is_available()
    if args.amp_dtype is None:
        args.amp_dtype = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    args.hardware = hardware
    return args


def main():
    args = parse_args()
    if DEVICE != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is required for baseline training, but the current runtime did not expose a usable GPU. "
            "Baseline execution has been blocked to avoid accidental CPU training.\n"
            f"Debug info: {format_cuda_debug_info()}\n"
            "If you really want to run on CPU for debugging only, pass --allow-cpu true."
        )

    setup_seed(args.seed)
    results_base = Path(args.results_dir) if args.results_dir else ROOT / "results/baselines"

    print(f"\n{'='*70}")
    print(f"Model: {args.model.upper()}  |  Dataset: {args.dataset}")
    print(f"Horizons: {args.horizon}  |  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(
            f"GPU: {args.hardware['gpu_name']}  |  Mem: {args.hardware['gpu_mem_gb']:.1f} GB  |  "
            f"Batch: {args.batch_size}  |  Workers: {args.num_workers}  |  "
            f"AMP: {args.use_amp}/{args.amp_dtype}"
        )
    print(f"{'='*70}")

    interrupted = False
    try:
        for h in args.horizon:
            out_dir = results_base / args.model / args.dataset / "72" / str(h)
            print(f"\n── Horizon {h}h ──────────────────────────────────────────────────")
            train_one_horizon(args.model, args.dataset, h, args, out_dir)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⚠️  Interrupted by user (Ctrl+C). Cleaning up CUDA resources...")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if interrupted:
            print("  CUDA cache released.")

    print(f"\n✅ Done. Results: {results_base / args.model / args.dataset}/")


if __name__ == "__main__":
    main()
