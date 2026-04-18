"""
Helpers for the lightweight public dataset layout.

The public repository keeps compact continuous station panels under
`dataset/{name}/AQI_processed/` and regenerates large sliding-window NPZ files
locally when needed. This module centralizes:

1. locating compressed or uncompressed panel files,
2. reading station panels with stable ordering,
3. rebuilding `train_val_test_data/72_{h}/...` from the compact panels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FEATURE_ORDER = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_deg",
]


def _strip_csv_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".csv.gz"):
        return name[:-7]
    if name.endswith(".csv"):
        return name[:-4]
    return path.stem


def _numeric_file_sort_key(path: Path) -> tuple[int, str]:
    stem = _strip_csv_suffix(path)
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1)), stem
    return 10**9, stem


def read_panel_csv(path: Path) -> pd.DataFrame:
    """Read one compact station panel from either `.csv` or `.csv.gz`."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def find_panel_paths(dataset_root: Path) -> list[Path]:
    """
    Return station panel files in deterministic order.

    Ordering rules:
    - Delhi datasets follow `location.csv` station_id order.
    - Other datasets follow the numeric PRSA_Data_{i} order.
    """
    aqi_dir = dataset_root / "AQI_processed"
    csvs = sorted(aqi_dir.glob("*.csv")) + sorted(aqi_dir.glob("*.csv.gz"))
    if not csvs:
        raise FileNotFoundError(f"No AQI_processed panel found under {aqi_dir}")

    csv_map = {_strip_csv_suffix(p): p for p in csvs}
    location_csv = dataset_root / "location" / "location.csv"
    if location_csv.exists():
        loc = pd.read_csv(location_csv)
        if "station_id" in loc.columns:
            ordered = []
            for sid in loc["station_id"].astype(str).tolist():
                if sid in csv_map:
                    ordered.append(csv_map[sid])
            if ordered and len(ordered) == len(csvs):
                return ordered

    return sorted(csvs, key=_numeric_file_sort_key)


def _safe_fill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill compact station panels deterministically for split generation."""
    filled = df.copy()
    for col in filled.columns:
        series = pd.to_numeric(filled[col], errors="coerce")
        series = series.interpolate(method="time", limit_direction="both")
        series = series.ffill().bfill()
        if series.isna().any():
            mean_val = float(series.mean()) if not np.isnan(series.mean()) else 0.0
            series = series.fillna(mean_val)
        filled[col] = series.astype(np.float32)
    return filled


def _impute_missing_split(df_station: pd.DataFrame, interp_limit_short: int = 6) -> pd.DataFrame:
    """
    Station-wise imputation used by the CN-city paper protocol.

    This follows the paper-aligned builder closely:
    - local linear interpolation for short gaps,
    - same-month same-hour fallback afterwards.
    """
    df = df_station.copy()
    df = df.interpolate(method="linear", limit=interp_limit_short, limit_direction="both")
    dt_index = df.index
    for col in df.columns:
        null_mask = df[col].isna()
        if null_mask.any():
            hour_month_mean = (
                df.loc[~null_mask, col]
                .groupby([dt_index[~null_mask].month, dt_index[~null_mask].hour])
                .mean()
            )
            for ts in df.index[null_mask]:
                key = (ts.month, ts.hour)
                if key in hour_month_mean.index:
                    df.loc[ts, col] = hour_month_mean[key]
    return df


def _stack_station_frames(frames: list[pd.DataFrame]) -> np.ndarray:
    arrays = []
    for frame in frames:
        frame = frame.copy()
        for col in FEATURE_ORDER:
            if col not in frame.columns:
                frame[col] = np.nan
        arrays.append(frame[FEATURE_ORDER].to_numpy(dtype=np.float32))
    return np.stack(arrays, axis=1).astype(np.float32)  # (T, N, F)


def _make_sliding_windows(stacked_norm: np.ndarray, seq_len: int, pred_len: int, target_idx: int = 0):
    valid_mask = np.all(np.isfinite(stacked_norm), axis=(1, 2))
    total = stacked_norm.shape[0] - seq_len - pred_len + 1
    x_list, y_list = [], []
    for start in range(total):
        if not np.all(valid_mask[start:start + seq_len + pred_len]):
            continue
        x = stacked_norm[start:start + seq_len]
        y = stacked_norm[start + seq_len:start + seq_len + pred_len, :, target_idx]
        x_list.append(x.transpose(1, 0, 2))
        y_list.append(y.T)
    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


def _save_npz_splits(out_dir: Path, splits: dict[str, tuple[np.ndarray, np.ndarray]], scaler: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (x, y) in splits.items():
        np.savez_compressed(out_dir / f"{split_name}_PM25.npz", X=x, y=y)
    np.save(out_dir / "scaler_PM25.npy", scaler.astype(np.float32))


def _load_manifest(dataset_root: Path) -> dict:
    path = dataset_root / "split_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_full_timeline_fraction(dataset_root: Path, manifest: dict, horizons: Iterable[int], seq_len: int, force: bool) -> None:
    frames = [read_panel_csv(p) for p in find_panel_paths(dataset_root)]
    frames = [_safe_fill_frame(df) for df in frames]
    stacked = _stack_station_frames(frames)

    target_idx = FEATURE_ORDER.index("PM2.5")
    flat = stacked.reshape(-1, stacked.shape[-1])
    feat_min = np.nanmin(flat, axis=0)
    feat_max = np.nanmax(flat, axis=0)
    scale = feat_max - feat_min
    scale[scale == 0] = 1.0
    stacked_norm = (stacked - feat_min) / scale
    scaler = np.array([feat_min[target_idx], feat_max[target_idx]], dtype=np.float32)

    fractions = manifest.get("split_fraction", [0.6, 0.2, 0.2])
    train_frac, val_frac, _ = fractions

    for pred_len in horizons:
        out_dir = dataset_root / "train_val_test_data" / f"{seq_len}_{pred_len}"
        train_npz = out_dir / "train_PM25.npz"
        if train_npz.exists() and not force:
            continue
        x, y = _make_sliding_windows(stacked_norm, seq_len, pred_len, target_idx)
        n = len(x)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits = {
            "train": (x[:n_train], y[:n_train]),
            "val": (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            "test": (x[n_train + n_val:], y[n_train + n_val:]),
        }
        _save_npz_splits(out_dir, splits, scaler)


def _build_absolute_time_train_scaler(dataset_root: Path, manifest: dict, horizons: Iterable[int], seq_len: int, force: bool) -> None:
    panel_paths = find_panel_paths(dataset_root)
    frames = [read_panel_csv(p) for p in panel_paths]
    full_index = frames[0].index

    split_frames = {}
    for split_name in ("train", "val", "test"):
        start = manifest[split_name]["start"]
        end = manifest[split_name]["end"]
        split_frames[split_name] = [df.loc[start:end].copy() for df in frames]

    filled = {}
    for split_name, station_frames in split_frames.items():
        filled[split_name] = [_impute_missing_split(frame[FEATURE_ORDER]) for frame in station_frames]

    target_idx = FEATURE_ORDER.index("PM2.5")
    train_stacked = _stack_station_frames(filled["train"])
    flat_train = train_stacked.reshape(-1, train_stacked.shape[-1])
    feat_min = np.nanmin(flat_train, axis=0)
    feat_max = np.nanmax(flat_train, axis=0)
    scale = feat_max - feat_min
    scale[scale == 0] = 1.0
    scaler = np.array([feat_min[target_idx], feat_max[target_idx]], dtype=np.float32)

    stacked_norm = {}
    for split_name in ("train", "val", "test"):
        arr = _stack_station_frames(filled[split_name])
        stacked_norm[split_name] = (arr - feat_min) / scale

    for pred_len in horizons:
        out_dir = dataset_root / "train_val_test_data" / f"{seq_len}_{pred_len}"
        train_npz = out_dir / "train_PM25.npz"
        if train_npz.exists() and not force:
            continue
        splits = {}
        for split_name in ("train", "val", "test"):
            splits[split_name] = _make_sliding_windows(stacked_norm[split_name], seq_len, pred_len, target_idx)
        _save_npz_splits(out_dir, splits, scaler)


def build_dataset_splits(dataset: str, horizons: Iterable[int] | None = None, seq_len: int = 72, force: bool = False) -> Path:
    dataset_root = ROOT / "dataset" / dataset
    manifest = _load_manifest(dataset_root)
    target_horizons = list(horizons or manifest.get("pred_lens", [1, 6, 12, 24]))
    split_mode = manifest.get("split_mode")

    if split_mode == "window_fraction_full_timeline":
        _build_full_timeline_fraction(dataset_root, manifest, target_horizons, seq_len, force)
    elif split_mode == "absolute_time_split_train_scaler":
        _build_absolute_time_train_scaler(dataset_root, manifest, target_horizons, seq_len, force)
    else:
        raise ValueError(f"Unsupported split_mode for {dataset}: {split_mode!r}")

    return dataset_root / "train_val_test_data"


def ensure_prepared_splits(dataset: str, horizons: Iterable[int] | None = None, seq_len: int = 72) -> None:
    dataset_root = ROOT / "dataset" / dataset
    manifest = _load_manifest(dataset_root)
    target_horizons = list(horizons or manifest.get("pred_lens", [1, 6, 12, 24]))
    missing = []
    for pred_len in target_horizons:
        split_dir = dataset_root / "train_val_test_data" / f"{seq_len}_{pred_len}"
        if not (
            (split_dir / "train_PM25.npz").exists()
            and (split_dir / "val_PM25.npz").exists()
            and (split_dir / "test_PM25.npz").exists()
            and (split_dir / "scaler_PM25.npy").exists()
        ):
            missing.append(pred_len)
    if missing:
        build_dataset_splits(dataset, missing, seq_len=seq_len, force=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper NPZ splits from compact AQI_processed station panels")
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset name(s) or the keyword 'all'")
    parser.add_argument("--pred-lens", nargs="+", type=int, default=None, help="Subset of horizons to build")
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--force", action="store_true", help="Overwrite existing NPZ split files")
    args = parser.parse_args()

    datasets = args.dataset
    if datasets == ["all"]:
        datasets = [
            "Beijing_12",
            "Beijing_Recent_12",
            "Chengdu_10",
            "Delhi_NCT_Meteo",
        ]

    for dataset in datasets:
        out_dir = build_dataset_splits(dataset, horizons=args.pred_lens, seq_len=args.seq_len, force=args.force)
        print(f"Prepared NPZ splits for {dataset}: {out_dir}")


if __name__ == "__main__":
    main()
