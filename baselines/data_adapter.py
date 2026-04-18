"""
baselines/data_adapter.py
=========================
Local data-adaptation layer for the baseline models.

Goals:
1. Reuse the unified NPZ splits and PM2.5 scaler produced by this project.
2. Reconstruct covariates according to each model's original input design,
   instead of forcing every baseline into one simplified tensor format.
3. Keep all datasets comparable under the same train/validation/test protocol.

Currently supported:
  - AGCRN:        PM2.5 only, x shape (B, T, N, 1)
  - iTransformer: PM2.5 multivariate series + calendar covariates
                  x_enc  shape (B, T, N)
                  x_mark shape (B, T, 4) = [month, day, weekday, hour] (normalized)
  - STAEformer:   PM2.5 + TOD + DOW
                  x shape (B, T, N, 3)
  - PM2.5-GNN:    official-style local adaptation
                  pm25_hist shape (B, T, N, 1)
                  feature   shape (B, T+H, N, F_exog)

Notes:
  - Chinese-city datasets recover the absolute-time split from
    `split_manifest.json`.
  - Delhi_NCT keeps its legacy 6:2:2 window split, so time covariates are
    reconstructed by first generating the full window stream from the
    continuous AQI_processed timeline and then slicing by sample count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.compact_dataset import ensure_prepared_splits, find_panel_paths

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAMES = {"agcrn", "itransformer", "staeformer", "pm25_gnn", "mstgan", "airformer"}
DELHI_DATASET = "Delhi_NCT_Meteo"

# Fixed input-feature protocol for MSTGAN and AirFormer.
# Within this repository the NPZ feature order is fixed as follows:
#   CN cities: [PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, WD_deg]
#   Delhi:     [PM2.5, PM10, SO2, NO2, CO, O3, NO, NOx, NH3, Benzene, Toluene, Xylene]
# The first six channels always represent the same air-quality attributes.
# This matches AirFormer's stochastic reconstruction logic (`X_rec[..., :6]`)
# and also matches MSTGAN's original six-feature Beijing setup.
MSTGAN_AIR_QUALITY_IDX = (0, 1, 2, 3, 4, 5)   # PM2.5 / PM10 / SO2 / NO2 / CO / O3
AIRFORMER_FEATURE_COUNT = 12                   # full continuous exogenous set
AIRFORMER_REC_CHANNELS = 6                     # reconstruct only first 6 (air quality)


def _load_npz(dataset: str, horizon: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "dataset" / dataset / "train_val_test_data" / f"72_{horizon}" / f"{split}_PM25.npz"
    if not path.exists():
        ensure_prepared_splits(dataset, horizons=[horizon], seq_len=72)
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


def _load_manifest(dataset: str) -> dict | None:
    path = ROOT / "dataset" / dataset / "split_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _full_day_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(
        start=f"{start} 00:00:00",
        end=f"{end} 23:00:00",
        freq="h",
    )


def _load_delhi_full_index(dataset: str) -> pd.DatetimeIndex:
    csvs = find_panel_paths(ROOT / "dataset" / dataset)
    df = pd.read_csv(csvs[0], index_col=0, parse_dates=True)
    return pd.DatetimeIndex(df.index)


def _load_full_index_from_aqi_processed(dataset: str) -> pd.DatetimeIndex:
    csvs = find_panel_paths(ROOT / "dataset" / dataset)
    df = pd.read_csv(csvs[0], index_col=0, parse_dates=True)
    return pd.DatetimeIndex(df.index)


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def _load_location_frame(dataset: str) -> pd.DataFrame:
    path = ROOT / "dataset" / dataset / "location" / "location.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing location file: {path}")
    return pd.read_csv(path)


def _load_ordered_station_frames(dataset: str) -> tuple[pd.DatetimeIndex, list[str], list[pd.DataFrame]]:
    aqi_dir = ROOT / "dataset" / dataset / "AQI_processed"
    csvs = find_panel_paths(ROOT / "dataset" / dataset)

    if dataset == DELHI_DATASET:
        location = _load_location_frame(dataset)
        ordered_names = location["station_id"].astype(str).tolist()
        frame_map = {p.stem: _read_indexed_csv(p) for p in csvs}
        frames = [frame_map[name] for name in ordered_names]
        full_index = frames[0].index
        return full_index, ordered_names, frames

    frames = [_read_indexed_csv(p) for p in csvs]
    full_index = frames[0].index
    ordered_names = [p.stem for p in csvs]
    return full_index, ordered_names, frames


def _safe_fill_frame(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for col in filled.columns:
        series = pd.to_numeric(filled[col], errors="coerce")
        series = series.interpolate(method="time", limit_direction="both")
        series = series.ffill().bfill()
        if series.isna().any():
            series = series.fillna(float(series.mean()) if not np.isnan(series.mean()) else 0.0)
        filled[col] = series.astype(np.float32)
    return filled


def _pm25_gnn_feature_spec(dataset: str, frame: pd.DataFrame) -> dict:
    columns = set(frame.columns)
    weather_cols = {"TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_deg"}
    if weather_cols.issubset(columns):
        return {
            "mode": "weather",
            "base_cols": ["TEMP", "PRES", "DEWP", "RAIN"],
            "wind_cols": ["WSPM", "WD_deg"],
            "use_wind": True,
        }

    base_cols = [c for c in ["PM10", "SO2", "NO2", "CO", "O3", "NO", "NOx", "NH3"] if c in columns]
    if not base_cols:
        raise ValueError(f"{dataset} has no supported PM2.5-GNN exogenous columns: {sorted(columns)}")
    return {
        "mode": "chemistry",
        "base_cols": base_cols,
        "wind_cols": [],
        "use_wind": False,
    }


def _build_pm25_gnn_feature_array(
    dataset: str,
    full_index: pd.DatetimeIndex,
    station_frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, dict]:
    station_frames = [_safe_fill_frame(df) for df in station_frames]
    spec = _pm25_gnn_feature_spec(dataset, station_frames[0])

    pm25_arr = np.stack([df["PM2.5"].to_numpy(dtype=np.float32) for df in station_frames], axis=1)
    time_hour = np.repeat(full_index.hour.to_numpy(dtype=np.float32)[:, None], len(station_frames), axis=1)
    time_weekday = np.repeat(full_index.weekday.to_numpy(dtype=np.float32)[:, None], len(station_frames), axis=1)

    feature_parts = []
    dim_names = []
    for col in spec["base_cols"]:
        feature_parts.append(np.stack([df[col].to_numpy(dtype=np.float32) for df in station_frames], axis=1))
        dim_names.append(col)

    feature_parts.append(time_hour)
    feature_parts.append(time_weekday)
    dim_names.extend(["hour", "weekday"])

    if spec["use_wind"]:
        for col in spec["wind_cols"]:
            feature_parts.append(np.stack([df[col].to_numpy(dtype=np.float32) for df in station_frames], axis=1))
            dim_names.append(col)

    feature_arr = np.stack(feature_parts, axis=-1).astype(np.float32)  # (T, N, F)
    meta = {
        "feature_mode": spec["mode"],
        "feature_dim": feature_arr.shape[-1],
        "dim_names": dim_names,
        "use_wind": spec["use_wind"],
    }
    return pm25_arr.astype(np.float32), feature_arr, meta


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lam = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(max(1e-12, 1.0 - a))))


def _bearing_radians(lat1, lon1, lat2, lon2) -> float:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_lam = np.radians(lon2 - lon1)
    y = np.sin(d_lam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lam)
    return float(np.arctan2(y, x))


def _build_pm25_gnn_graph(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    location = _load_location_frame(dataset)
    lat_col = "Latitude" if "Latitude" in location.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in location.columns else "longitude"
    coords = location[[lat_col, lon_col]].to_numpy(dtype=np.float32)

    edges = []
    attrs = []
    n_nodes = len(coords)
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst:
                continue
            lat1, lon1 = coords[src]
            lat2, lon2 = coords[dst]
            dist_km = max(_haversine_km(lat1, lon1, lat2, lon2), 1e-3)
            direction = _bearing_radians(lat1, lon1, lat2, lon2)
            edges.append([src, dst])
            attrs.append([dist_km, direction])
    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_attr = np.asarray(attrs, dtype=np.float32)
    return edge_index, edge_attr


def _infer_split_start_times(dataset: str, split: str, horizon: int, n_samples: int, seq_len: int) -> pd.DatetimeIndex:
    """
    Recover the input-sequence start time for each window sample.

    China datasets:
      preprocessing first applies absolute-time splits and then creates windows
      inside each split, so window start times come directly from the split's
      hourly index.

    Delhi:
      preprocessing first generates windows over the full sequence and only then
      applies the 6:2:2 split, so we reconstruct the global window starts
      before slicing into train/val/test segments.
    """
    manifest = _load_manifest(dataset)
    if manifest is not None:
        split_index = _full_day_hourly_index(manifest[split]["start"], manifest[split]["end"])
        expected = len(split_index) - seq_len - horizon + 1
        if expected != n_samples:
            raise ValueError(
                f"{dataset}/{split}/{horizon} expected {expected} windows from manifest, got {n_samples}"
            )
        return split_index[:n_samples]

    full_index = _load_full_index_from_aqi_processed(dataset)
    total_samples = len(full_index) - seq_len - horizon + 1
    n_train = int(total_samples * 0.6)
    n_val = int(total_samples * 0.2)
    n_test = total_samples - n_train - n_val
    expected_map = {"train": n_train, "val": n_val, "test": n_test}
    offset_map = {"train": 0, "val": n_train, "test": n_train + n_val}
    expected = expected_map[split]
    if expected != n_samples:
        raise ValueError(f"{dataset}/{split}/{horizon} expected {expected} windows, got {n_samples}")
    offset = offset_map[split]
    return full_index[offset: offset + n_samples]


def _build_time_window_features(start_times: pd.DatetimeIndex, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      marks_norm: (B, T, 4) float32  [month, day, weekday, hour] normalized to [0,1]
      tod:        (B, T)    float32  hour / 24
      dow:        (B, T)    int64    0..6
    """
    offsets = np.arange(seq_len, dtype="timedelta64[h]")
    starts = start_times.values.astype("datetime64[h]")[:, None]
    window_times = starts + offsets[None, :]
    dt = pd.DatetimeIndex(window_times.reshape(-1)).to_series()

    month = (dt.dt.month.to_numpy(dtype=np.float32).reshape(len(start_times), seq_len) - 1.0) / 11.0
    day = (dt.dt.day.to_numpy(dtype=np.float32).reshape(len(start_times), seq_len) - 1.0) / 30.0
    weekday = dt.dt.weekday.to_numpy(dtype=np.int64).reshape(len(start_times), seq_len)
    hour = dt.dt.hour.to_numpy(dtype=np.float32).reshape(len(start_times), seq_len)

    marks_norm = np.stack(
        [
            month,
            day,
            weekday.astype(np.float32) / 6.0,
            hour / 23.0,
        ],
        axis=-1,
    ).astype(np.float32)

    tod = (hour / 24.0).astype(np.float32)
    dow = weekday.astype(np.int64)
    return marks_norm, tod, dow


def load_scaler(dataset: str, horizon: int) -> tuple[float, float]:
    """
    Returns (min_val, range_val) from scaler_PM25.npy.
    The saved file stores [min, max] in current unified protocol.
    """
    path = ROOT / f"dataset/{dataset}/train_val_test_data/72_{horizon}/scaler_PM25.npy"
    sc = np.load(path).astype(np.float32)
    min_val = float(sc[0])
    max_val = float(sc[1])
    return min_val, max_val - min_val


def inverse_transform(y_scaled: np.ndarray, min_val: float, range_val: float) -> np.ndarray:
    return y_scaled * range_val + min_val


def inverse_transform_zscore(y_scaled: np.ndarray, mean_val: float, std_val: float) -> np.ndarray:
    return y_scaled * std_val + mean_val


def get_dataset_info(dataset: str) -> dict:
    path = ROOT / f"dataset/{dataset}/train_val_test_data/72_1/train_PM25.npz"
    data = np.load(path)
    _, n_nodes, seq_len, n_features = data["X"].shape
    return {"n_nodes": n_nodes, "seq_len": seq_len, "n_features_raw": n_features}


class OfficialBaselineDataset(Dataset):
    def __init__(self, dataset: str, split: str, horizon: int, model_name: str):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model_name: {model_name}")

        X_raw, y_raw = _load_npz(dataset, horizon, split)  # X=(B,N,T,F), y=(B,N,H)
        B, N, T, F = X_raw.shape

        # Common PM2.5-only view
        X_btnf = X_raw.transpose(0, 2, 1, 3)         # (B,T,N,F)
        pm = X_btnf[..., 0:1].astype(np.float32)      # (B,T,N,1)
        targets = y_raw.astype(np.float32)            # (B,N,H)

        start_times = _infer_split_start_times(dataset, split, horizon, B, T)
        marks_norm, tod, dow = _build_time_window_features(start_times, T)

        if model_name == "agcrn":
            # Official AGCRN defaults to input_dim=1, so we keep PM2.5 only.
            self.inputs = {"x": pm}
            self.targets = targets

        elif model_name == "itransformer":
            # Official iTransformer expects a multivariate target series plus x_mark.
            # We use PM2.5 from all N stations as the multivariate series.
            self.inputs = {
                "x_enc": pm[..., 0],           # (B,T,N)
                "x_mark_enc": marks_norm,      # (B,T,4)
            }
            self.targets = targets             # (B,N,H)

        elif model_name == "staeformer":
            # Official STAEformer expects [target, tod, dow].
            tod_broadcast = np.repeat(tod[:, :, None, None], N, axis=2)       # (B,T,N,1)
            dow_broadcast = np.repeat(dow[:, :, None, None], N, axis=2).astype(np.float32)
            x = np.concatenate([pm, tod_broadcast, dow_broadcast], axis=-1)    # (B,T,N,3)
            self.inputs = {"x": x}
            self.targets = targets[:, :, :].transpose(0, 2, 1)[..., None]      # (B,H,N,1)

        elif model_name == "mstgan":
            # Official input is (B, N, F, T), with F=6 for the Beijing setup.
            # We consistently use [PM2.5, PM10, SO2, NO2, CO, O3] so MSTGAN sees
            # the same feature semantics across all supported datasets.
            subset = X_raw[..., list(MSTGAN_AIR_QUALITY_IDX)]   # (B,N,T,6)
            x = subset.transpose(0, 1, 3, 2).astype(np.float32)  # (B,N,6,T)
            self.inputs = {"x": x}
            self.targets = targets              # (B,N,H)

        elif model_name == "airformer":
            # Official AirFormer input is (B, T, N, C), with main.py defaulting
            # to input_dim=27 (11 continuous + 16 embedded features). We do not
            # reproduce the categorical embedding branch here; instead we use the
            # 12 continuous features stored in the project NPZ files:
            #   first 6  = air-quality variables
            #   last 6   = meteorological / auxiliary variables
            feature_subset = X_btnf[..., :AIRFORMER_FEATURE_COUNT].astype(np.float32)
            self.inputs = {"x": feature_subset}  # (B,T,N,12)
            self.targets = targets              # (B,N,H)

        self.n_samples = B
        self.dataset = dataset
        self.split = split
        self.horizon = horizon
        self.model_name = model_name

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        batch = {k: torch.from_numpy(v[idx]) for k, v in self.inputs.items()}
        batch["y"] = torch.from_numpy(self.targets[idx])
        return batch


@dataclass
class BaselineMeta:
    inverse_kind: str
    pm25_min: float | None = None
    pm25_range: float | None = None
    pm25_mean: float | None = None
    pm25_std: float | None = None
    feature_dim: int | None = None
    edge_index: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    use_wind: bool | None = None
    dim_names: list[str] | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    # MSTGAN: Chebyshev polynomials from distance adjacency, each (N, N)
    cheb_polynomials: list[np.ndarray] | None = None
    mstgan_input_dim: int | None = None
    # AirFormer: dartboard tensors + multivariate input dim
    dartboard_assignment: np.ndarray | None = None
    dartboard_mask: np.ndarray | None = None
    dartboard_num_sectors: int | None = None
    dartboard_radii_km: tuple[float, float] | None = None
    airformer_input_dim: int | None = None
    airformer_rec_channels: int | None = None


class PM25GNNDataset(Dataset):
    def __init__(
        self,
        pm25_hist: np.ndarray,
        feature: np.ndarray,
        target: np.ndarray,
    ):
        self.pm25_hist = pm25_hist
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.pm25_hist)

    def __getitem__(self, idx: int):
        return {
            "pm25_hist": torch.from_numpy(self.pm25_hist[idx].astype(np.float32)),
            "feature": torch.from_numpy(self.feature[idx].astype(np.float32)),
            "y": torch.from_numpy(self.target[idx].astype(np.float32)),
        }


def _build_pm25_gnn_bundle(
    dataset: str,
    horizon: int,
    batch_size: int,
    num_workers: int,
    loader_config: dict | None = None,
):
    seq_len = 72
    loader_config = loader_config or {}
    pin_memory = loader_config.get("pin_memory", torch.cuda.is_available())
    persistent_workers = bool(loader_config.get("persistent_workers", num_workers > 0))
    prefetch_factor = loader_config.get("prefetch_factor", 2)
    timeout = int(loader_config.get("timeout", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (num_workers > 0 and persistent_workers),
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if timeout > 0:
        loader_kwargs["timeout"] = timeout

    if dataset == DELHI_DATASET:
        raw_feature_names = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NO", "NOx", "NH3", "Benzene", "Toluene", "Xylene"]
        base_feature_idx = [1, 2, 3, 4, 5, 6, 7, 8]
        use_wind = False
        dim_names = ["PM10", "SO2", "NO2", "CO", "O3", "NO", "NOx", "NH3", "hour", "weekday"]
    else:
        raw_feature_names = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_deg"]
        base_feature_idx = [6, 7, 8, 9, 10, 11]
        use_wind = True
        dim_names = ["TEMP", "PRES", "DEWP", "RAIN", "hour", "weekday", "WSPM", "WD_deg"]

    per_split = {}
    train_features_for_stats = []
    feature_dim = len(dim_names)

    for split in ["train", "val", "test"]:
        X_raw, y_raw = _load_npz(dataset, horizon, split)  # (B,N,T,F), (B,N,H)
        B, N, T, _ = X_raw.shape
        X = X_raw.transpose(0, 2, 1, 3)                    # (B,T,N,F)
        pm25_hist = X[..., 0:1].astype(np.float32)         # (B,T,N,1)
        base_feat = X[..., base_feature_idx].astype(np.float32)

        start_times = _infer_split_start_times(dataset, split, horizon, B, seq_len)
        full_marks, _, dow = _build_time_window_features(start_times, seq_len + horizon)
        hour_norm = full_marks[..., 3].astype(np.float32)            # (B,T+H)
        dow_norm = (dow.astype(np.float32) / 6.0)                    # (B,T+H)

        hist_hour = np.repeat(hour_norm[:, :T, None, None], N, axis=2)
        fut_hour = np.repeat(hour_norm[:, T:, None, None], N, axis=2)
        hist_dow = np.repeat(dow_norm[:, :T, None, None], N, axis=2)
        fut_dow = np.repeat(dow_norm[:, T:, None, None], N, axis=2)

        hist_base = base_feat
        fut_base = np.repeat(base_feat[:, -1:, :, :], horizon, axis=1)

        if use_wind:
            hist_parts = [hist_base[..., :4], hist_hour, hist_dow, hist_base[..., 4:]]
            fut_parts = [fut_base[..., :4], fut_hour, fut_dow, fut_base[..., 4:]]
        else:
            hist_parts = [hist_base, hist_hour, hist_dow]
            fut_parts = [fut_base, fut_hour, fut_dow]

        feature_window = np.concatenate(
            [
                np.concatenate(hist_parts, axis=-1),
                np.concatenate(fut_parts, axis=-1),
            ],
            axis=1,
        ).astype(np.float32)                                        # (B,T+H,N,F_exog)

        per_split[split] = {
            "pm25_hist": pm25_hist,
            "feature_window": feature_window,
            "target": y_raw.astype(np.float32),
        }
        if split == "train":
            train_features_for_stats.append(feature_window)

    train_feature = train_features_for_stats[0]
    feature_mean = train_feature.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    feature_std = (train_feature.std(axis=(0, 1, 2), keepdims=True) + 1e-6).astype(np.float32)

    loaders = []
    for split in ["train", "val", "test"]:
        feature_window = (per_split[split]["feature_window"] - feature_mean) / feature_std
        ds = PM25GNNDataset(
            pm25_hist=per_split[split]["pm25_hist"],
            feature=feature_window.astype(np.float32),
            target=per_split[split]["target"],
        )
        loaders.append(
            DataLoader(
                ds,
                shuffle=(split == "train"),
                **loader_kwargs,
            )
        )

    edge_index, edge_attr = _build_pm25_gnn_graph(dataset)
    meta = BaselineMeta(
        inverse_kind="minmax",
        pm25_min=load_scaler(dataset, horizon)[0],
        pm25_range=load_scaler(dataset, horizon)[1],
        feature_dim=feature_dim,
        edge_index=edge_index,
        edge_attr=edge_attr,
        use_wind=use_wind,
        dim_names=dim_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    return tuple(loaders), meta


def _distance_adjacency(dataset: str, sigma: float | None = None) -> np.ndarray:
    """Build a symmetric Gaussian-kernel distance adjacency matrix (N, N)."""
    location = _load_location_frame(dataset)
    lat_col = "Latitude" if "Latitude" in location.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in location.columns else "longitude"
    coords = location[[lat_col, lon_col]].to_numpy(dtype=np.float32)
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist[i, j] = _haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
    if sigma is None:
        sigma = float(dist[dist > 0].std()) if (dist > 0).any() else 1.0
    W = np.exp(-(dist ** 2) / (2.0 * max(sigma, 1e-6) ** 2))
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def _scaled_laplacian(W: np.ndarray) -> np.ndarray:
    """Compute \tilde{L} = 2L / lambda_max - I (dense, for small N)."""
    from scipy.sparse.linalg import eigs

    D = np.diag(W.sum(axis=1))
    L = D - W
    lambda_max = eigs(L.astype(np.float64), k=1, which="LR")[0].real
    lam = float(lambda_max[0] if hasattr(lambda_max, "__len__") else lambda_max)
    if lam <= 0:
        lam = 2.0
    return ((2.0 * L) / lam - np.identity(W.shape[0])).astype(np.float32)


def _cheb_polynomials(L_tilde: np.ndarray, K: int) -> list[np.ndarray]:
    n = L_tilde.shape[0]
    polys = [np.identity(n, dtype=np.float32), L_tilde.astype(np.float32)]
    for k in range(2, K):
        polys.append((2.0 * L_tilde @ polys[-1] - polys[-2]).astype(np.float32))
    return polys[:K]


def _build_mstgan_meta(dataset: str, horizon: int, K: int = 3) -> dict:
    W = _distance_adjacency(dataset)
    L_tilde = _scaled_laplacian(W)
    polys = _cheb_polynomials(L_tilde, K)
    return {"cheb_polynomials": polys}


def _build_airformer_meta(dataset: str) -> dict:
    """
    Build dartboard assignment/mask for AirFormer.

    Rule (local adaptation, documented in baselines/airformer/model.py):
      Official AirFormer uses fixed (50, 200) km ring radii at a nationwide
      scale. On compact city-level station networks, that leaves many sectors
      empty and causes DS-MSA to degenerate. Here we adapt the rings by the
      network's maximum pairwise station distance:
          r_inner = 0.15 * max_pairwise_distance
          r_outer = 0.45 * max_pairwise_distance
      8 angular sectors, num_sectors = 1 (self) + 2*8 + 1 (far) = 18.
      This keeps the "ring + sector" design from the paper while avoiding the
      earlier piecewise hard-coded radii, and it works across all supported
      datasets.
    """
    from baselines.airformer.model import build_dartboard

    location = _load_location_frame(dataset)
    lat_col = "Latitude" if "Latitude" in location.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in location.columns else "longitude"
    coords = location[[lat_col, lon_col]].to_numpy(dtype=np.float32)

    n = len(coords)
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            max_dist = max(max_dist, d)
    max_dist = max(max_dist, 1.0)  # avoid zero-radius degeneration
    radii = (0.15 * max_dist, 0.45 * max_dist)
    assignment, mask, num_sectors = build_dartboard(coords, radii_km=radii, num_angles=8)
    return {
        "dartboard_assignment": assignment,
        "dartboard_mask": mask,
        "dartboard_num_sectors": num_sectors,
        "dartboard_radii_km": radii,
        "dartboard_max_dist_km": max_dist,
    }


def get_dataloaders(
    model_name: str,
    dataset: str,
    horizon: int,
    batch_size: int = 64,
    num_workers: int = 2,
    loader_config: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, BaselineMeta]:
    loader_config = loader_config or {}
    pin_memory = loader_config.get("pin_memory", torch.cuda.is_available())
    persistent_workers = bool(loader_config.get("persistent_workers", num_workers > 0))
    prefetch_factor = loader_config.get("prefetch_factor", 2)
    timeout = int(loader_config.get("timeout", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (num_workers > 0 and persistent_workers),
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if timeout > 0:
        loader_kwargs["timeout"] = timeout

    if model_name == "pm25_gnn":
        loaders, meta = _build_pm25_gnn_bundle(
            dataset,
            horizon,
            batch_size,
            num_workers,
            loader_config=loader_config,
        )
        return (*loaders, meta)

    loaders = []
    for split in ["train", "val", "test"]:
        ds = OfficialBaselineDataset(dataset=dataset, split=split, horizon=horizon, model_name=model_name)
        loaders.append(
            DataLoader(
                ds,
                shuffle=(split == "train"),
                **loader_kwargs,
            )
        )
    min_val, range_val = load_scaler(dataset, horizon)
    meta = BaselineMeta(inverse_kind="minmax", pm25_min=min_val, pm25_range=range_val)

    if model_name == "mstgan":
        # Keep the same channel definition as MSTGAN_AIR_QUALITY_IDX in
        # OfficialBaselineDataset: 6 air-quality channels.
        meta.mstgan_input_dim = len(MSTGAN_AIR_QUALITY_IDX)
        meta.cheb_polynomials = _build_mstgan_meta(dataset, horizon, K=3)["cheb_polynomials"]
    elif model_name == "airformer":
        ab = _build_airformer_meta(dataset)
        meta.dartboard_assignment = ab["dartboard_assignment"]
        meta.dartboard_mask = ab["dartboard_mask"]
        meta.dartboard_num_sectors = ab["dartboard_num_sectors"]
        meta.dartboard_radii_km = ab["dartboard_radii_km"]
        meta.airformer_input_dim = AIRFORMER_FEATURE_COUNT
        meta.airformer_rec_channels = AIRFORMER_REC_CHANNELS

    return (*loaders, meta)
