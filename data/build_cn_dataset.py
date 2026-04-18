"""
Build DMGENet training datasets from the national CN air-quality + ISD-Lite
meteorology archives.

Subcommands:
  audit      Data-quality audit only — writes a per-station coverage report.
  preprocess Full pipeline — writes NPZ splits + AQI_processed + location.csv.

Inputs (paths on the author's machine — override DATA_AQ / DATA_METEO below):
  Air quality:  /home/bm/Data/全国空气质量/站点_YYYYMMDD-YYYYMMDD.zip
  Meteorology:  /home/bm/Data/中国气象数据/china_isd_lite_YYYY.zip
  Station list: /home/bm/Data/全国空气质量/_站点列表/站点列表-2020.01.01起.csv

Typical use:
  python data/build_cn_dataset.py audit      --city chengdu
  python data/build_cn_dataset.py preprocess --city beijing_recent
  python data/build_cn_dataset.py preprocess --city chengdu
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Paths and per-city configuration
# ──────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent          # DMGENet/
DATA_AQ    = Path("/home/bm/Data/全国空气质量")
DATA_METEO = Path("/home/bm/Data/中国气象数据")
STATION_LIST_CSV = DATA_AQ / "_站点列表" / "站点列表-2020.01.01起.csv"

CITY_CONFIGS = {
    "beijing_recent": {
        "stations": [
            "1001A","1002A","1003A","1004A","1005A","1006A",
            "1007A","1008A","1009A","1010A","1011A","1012A",
        ],
        "meteo_station": "545110-99999",
        "meteo_altitude_m": 31.3,
        "pres_strategy": "slp_correction",
        "dataset_name": "Beijing_Recent_12",
        "train_start": "2020-01-01", "train_end": "2021-12-31",
        "val_start":   "2022-01-01", "val_end":   "2022-12-31",
        "test_start":  "2023-01-01", "test_end":  "2023-12-31",
    },
    "chengdu": {
        "stations": [
            "1431A","1432A","1433A","1434A","1435A",
            "1436A","1437A","1438A","2880A","3136A",
        ],
        "meteo_station": "562940-99999",
        "meteo_altitude_m": 495.3,
        # ISD-Lite SLP is ~0% populated for this station, so we substitute a
        # climatological mean at the station's altitude with seasonal offsets.
        "pres_strategy": "climate_constant",
        "pres_climate_mean": 966.0,    # ≈ 1013 − 495 × 0.1197 hPa, rounded
        "pres_climate_std":  8.0,      # typical seasonal range
        "dataset_name": "Chengdu_10",
        "train_start": "2020-01-01", "train_end": "2021-12-31",
        "val_start":   "2022-01-01", "val_end":   "2022-12-31",
        "test_start":  "2023-01-01", "test_end":  "2023-12-31",
    },
}

# The six pollutants and the final 12-feature order (matches Beijing_12).
AQ_TYPES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
FEATURE_ORDER = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                 "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_deg"]

SEQ_LEN   = 72
PRED_LENS = [1, 6, 12, 24]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: load national air-quality station data
# ──────────────────────────────────────────────────────────────────────────────
def _aq_zip_path(year: int) -> Path:
    """Return the per-year AQ zip (tolerates the non-standard 2014 naming)."""
    for name in DATA_AQ.glob(f"站点_{year}*"):
        if name.suffix == ".zip":
            return name
    raise FileNotFoundError(f"No AQ zip found for year {year} in {DATA_AQ}")


def _load_aq_day(zip_obj: zipfile.ZipFile, date: pd.Timestamp, stations: list) -> pd.DataFrame:
    """Read one day from an already-open year zip.

    Returns a DataFrame with a (hour, station) index and one column per
    pollutant in AQ_TYPES. Filters out macOS resource-fork files (`._…`).
    """
    day_str = date.strftime("%Y%m%d")
    matches = [n for n in zip_obj.namelist()
               if f"china_sites_{day_str}.csv" in n and not n.split("/")[-1].startswith("._")]
    if not matches:
        return None
    with zip_obj.open(matches[0]) as f:
        raw = pd.read_csv(f, low_memory=False)

    avail_stations = [s for s in stations if s in raw.columns]
    rows = []
    for aq_type in AQ_TYPES:
        sub = raw[raw["type"] == aq_type]
        if sub.empty:
            continue
        for s in avail_stations:
            for h, v in enumerate(sub[s].values):
                rows.append({"hour": h, "station": s, "type": aq_type, "value": v})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["hour", "station"], columns="type", values="value", aggfunc="first")
    return piv.reindex(columns=AQ_TYPES)


def load_aq_data(stations: list, start: str, end: str) -> pd.DataFrame:
    """Load hourly AQ data for `stations` across [start, end].

    Returns a MultiIndex DataFrame indexed by (datetime_CST, station) with
    one column per pollutant. Native timestamps are already CST (UTC+8).
    """
    date_range = pd.date_range(start, end, freq="D")
    years = sorted(set(d.year for d in date_range))
    print(f"  Loading AQ data: {start} ~ {end} ({len(date_range)} days across {years})")

    all_frames = []
    for year in years:
        zip_path = _aq_zip_path(year)
        print(f"    opening {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for date in date_range:
                if date.year != year:
                    continue
                day_df = _load_aq_day(zf, date, stations)
                if day_df is None:
                    continue
                day_df = day_df.reset_index()
                day_df["datetime"] = (
                    pd.to_datetime(date.strftime("%Y-%m-%d"))
                    + pd.to_timedelta(day_df["hour"], unit="h")
                )
                day_df = (day_df.set_index(["datetime", "station"])
                                .drop(columns=["hour"]))
                all_frames.append(day_df)

    if not all_frames:
        raise ValueError("No AQ records loaded — check data paths and station codes.")

    aq_df = pd.concat(all_frames).sort_index()
    n_stations = aq_df.index.get_level_values("station").nunique()
    print(f"  AQ loaded: {len(aq_df)} rows across {n_stations} stations")
    return aq_df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: load ISD-Lite meteorology
# ──────────────────────────────────────────────────────────────────────────────
def _isd_zip_path(year: int) -> Path:
    for p in DATA_METEO.glob(f"china_isd_lite_{year}*.zip"):
        return p
    raise FileNotFoundError(f"No ISD-Lite zip found for year {year} in {DATA_METEO}")


def _parse_isd_file(lines: list) -> pd.DataFrame:
    """Parse ISD-Lite fixed-width records into a UTC-indexed DataFrame."""
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            year, mo, da, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

            def fv(s):
                v = int(s)
                return np.nan if v == -9999 else v

            rec = {
                "TEMP": fv(parts[4]) / 10.0 if fv(parts[4]) is not np.nan else np.nan,
                "DEWP": fv(parts[5]) / 10.0 if fv(parts[5]) is not np.nan else np.nan,
                "SLP":  fv(parts[6]) / 10.0 if fv(parts[6]) is not np.nan else np.nan,
                "WDIR": fv(parts[7]),
                "WSPD": fv(parts[8]) / 10.0 if len(parts) > 8 and fv(parts[8]) is not np.nan else np.nan,
                "PREC1H": np.nan,
            }
            # PREC1H at index 10: -9999 = missing (treat as 0 per ISD §11.3),
            # -1 = trace precipitation (treat as 0.1 mm).
            if len(parts) > 10:
                raw_p = int(parts[10])
                if raw_p == -9999:
                    rec["PREC1H"] = 0.0
                elif raw_p == -1:
                    rec["PREC1H"] = 0.1
                else:
                    rec["PREC1H"] = raw_p / 10.0
            else:
                rec["PREC1H"] = 0.0
            rec["datetime_utc"] = pd.Timestamp(year=year, month=mo, day=da, hour=hr, tz="UTC")
            records.append(rec)
        except (ValueError, IndexError):
            continue

    return pd.DataFrame(records).set_index("datetime_utc")


def load_meteo_data(meteo_station: str, start: str, end: str,
                    altitude_m: float, pres_strategy: str,
                    pres_climate_mean: float = 966.0,
                    pres_climate_std: float = 8.0) -> pd.DataFrame:
    """Load ISD-Lite meteorology for a single station and return a CST-indexed
    DataFrame with the six columns: TEMP, PRES, DEWP, RAIN, WSPM, WD_deg.
    """
    date_range = pd.date_range(start, end, freq="D")
    years = sorted(set(d.year for d in date_range))
    print(f"  Loading meteo: station {meteo_station}, years {years}")

    all_frames = []
    for year in years:
        zip_path = _isd_zip_path(year)
        with zipfile.ZipFile(zip_path, "r") as zf:
            matches = [n for n in zf.namelist() if meteo_station in n and str(year) in n]
            if not matches:
                print(f"    [warn] {year}: station {meteo_station} missing, will NaN-fill")
                continue
            with zf.open(matches[0]) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
            all_frames.append(_parse_isd_file(lines))

    if not all_frames:
        raise ValueError(f"Station {meteo_station} has no data in {years}")

    meteo_raw = pd.concat(all_frames).sort_index()

    # UTC → CST (+8h) and drop tz for compatibility with AQ timestamps.
    meteo_raw.index = meteo_raw.index.tz_convert("Asia/Shanghai").tz_localize(None)
    meteo_raw = meteo_raw.loc[start:end]

    # Reindex to a complete hourly grid so gaps become explicit.
    full_idx = pd.date_range(start, end + " 23:00:00", freq="h")
    meteo_raw = meteo_raw.reindex(full_idx)

    # PRES: either correct SLP down to station elevation, or fall back to a
    # month-varying climatological mean when SLP is unavailable (Chengdu).
    slp_corr = altitude_m * 0.1197
    if pres_strategy == "slp_correction":
        meteo_raw["PRES"] = meteo_raw["SLP"] - slp_corr
        meteo_raw["PRES"] = meteo_raw["PRES"].interpolate(method="linear", limit=24)
    elif pres_strategy == "climate_constant":
        # Seasonal offsets: Chinese interior cities — highest in winter, lowest in summer.
        month_offsets = {1: 6, 2: 5, 3: 3, 4: 1, 5: -1, 6: -4,
                         7: -6, 8: -5, 9: -3, 10: 0, 11: 3, 12: 5}
        months = pd.DatetimeIndex(full_idx).month
        meteo_raw["PRES"] = np.array(
            [pres_climate_mean + month_offsets.get(m, 0) for m in months],
            dtype=np.float32,
        )

    meteo_raw["RAIN"] = meteo_raw["PREC1H"].fillna(0.0)

    meteo_raw["TEMP"] = meteo_raw["TEMP"].interpolate(method="linear", limit=6)
    meteo_raw["DEWP"] = meteo_raw["DEWP"].interpolate(method="linear", limit=6)

    meteo_raw["WSPM"] = meteo_raw["WSPD"].interpolate(method="linear", limit=6)
    meteo_raw["WSPM"] = meteo_raw["WSPM"].fillna(meteo_raw["WSPM"].median())

    # Wind direction is circular — interpolate sin/cos separately, then recombine.
    # Linear interpolation of raw degrees would wrap 359°→1° through 180°.
    wdir = meteo_raw["WDIR"].copy()
    valid = wdir.notna()
    if valid.sum() > 0:
        rad = np.deg2rad(wdir[valid])
        sin_w = pd.Series(np.sin(rad).values, index=wdir[valid].index).reindex(full_idx)
        cos_w = pd.Series(np.cos(rad).values, index=wdir[valid].index).reindex(full_idx)
        sin_w = sin_w.interpolate(method="linear", limit=6).ffill().bfill()
        cos_w = cos_w.interpolate(method="linear", limit=6).ffill().bfill()
        meteo_raw["WD_deg"] = np.rad2deg(np.arctan2(sin_w, cos_w)) % 360
    else:
        meteo_raw["WD_deg"] = 180.0

    meteo_out = meteo_raw[["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_deg"]].copy()
    print(f"  Meteo loaded: {len(meteo_out)} hours")
    _print_meteo_coverage(meteo_out)
    return meteo_out


def _print_meteo_coverage(df: pd.DataFrame):
    for col in df.columns:
        valid = df[col].notna().sum()
        total = len(df)
        print(f"    {col}: {valid}/{total} ({valid/total*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: merge AQ × meteorology into a 12-feature hourly panel
# ──────────────────────────────────────────────────────────────────────────────
def build_merged_table(aq_df: pd.DataFrame, meteo_df: pd.DataFrame,
                       stations: list, start: str, end: str) -> pd.DataFrame:
    """Merge multi-station AQ with the single meteo series.

    Station order is preserved from the `stations` argument so the panel's
    row order matches location.csv.
    """
    print("  Merging AQ + meteo...")
    full_idx = pd.date_range(start, end + " 23:00:00", freq="h")

    merged_frames = []
    aq_stations = set(aq_df.index.get_level_values("station"))
    for station in stations:
        if station in aq_stations:
            station_aq = aq_df.xs(station, level="station").reindex(full_idx)
        else:
            print(f"  [warn] station {station}: no AQ data, filling NaN")
            station_aq = pd.DataFrame(np.nan, index=full_idx, columns=AQ_TYPES)

        station_combined = pd.concat([station_aq, meteo_df], axis=1)
        station_combined.index.name = "datetime"
        station_combined["station"] = station
        station_combined = station_combined.reset_index().set_index(["datetime", "station"])
        merged_frames.append(station_combined[FEATURE_ORDER])

    merged = pd.concat(merged_frames).sort_index()
    print(f"  merged shape: {merged.shape}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: tiered missing-value handling + coverage audit
# ──────────────────────────────────────────────────────────────────────────────
def _max_consecutive_nan(series: pd.Series) -> int:
    max_run = run = 0
    for v in series.isna():
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def impute_missing(df_station: pd.DataFrame, interp_limit_short: int = 6,
                   interp_limit_medium: int = 24) -> pd.DataFrame:
    """Tiered missing-value imputation per station.

      ≤ 6h  : linear interpolation
      6-24h : linear + same-month-same-hour historical mean fallback
      > 24h : left as NaN so `make_sliding_windows` drops the affected windows
              rather than feeding invented values into the model.
    """
    df = df_station.copy()
    df = df.interpolate(method="linear", limit=interp_limit_short, limit_direction="both")

    df_idx = df.index
    for col in df.columns:
        null_mask = df[col].isna()
        if null_mask.any():
            hour_month_mean = (
                df.loc[~null_mask, col]
                .groupby([df_idx[~null_mask].month, df_idx[~null_mask].hour])
                .mean()
            )
            for i in df.index[null_mask]:
                key = (i.month, i.hour)
                if key in hour_month_mean.index:
                    df.loc[i, col] = hour_month_mean[key]
    return df


def audit_coverage(merged: pd.DataFrame, stations: list) -> pd.DataFrame:
    rows = []
    for station in stations:
        try:
            st_df = merged.xs(station, level="station")
        except KeyError:
            continue
        for col in FEATURE_ORDER:
            total = len(st_df)
            valid = st_df[col].notna().sum() if col in st_df.columns else 0
            max_gap = _max_consecutive_nan(st_df[col]) if col in st_df.columns else total
            rows.append({
                "station": station, "feature": col,
                "total_hours": total, "valid_hours": int(valid),
                "coverage_pct": round(valid / total * 100, 2) if total > 0 else 0,
                "max_gap_hours": int(max_gap),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: time splits + train-only normalisation + sliding windows
# ──────────────────────────────────────────────────────────────────────────────
def build_splits(merged: pd.DataFrame, stations: list,
                 train_start: str, train_end: str,
                 val_start: str,   val_end: str,
                 test_start: str,  test_end: str):
    """Slice by absolute dates (not ratios) and reshape to a wide table."""
    wide = merged.unstack(level="station")
    wide.columns = [f"{feat}_{sta}" for feat, sta in wide.columns]

    splits_raw = {
        "train": wide.loc[train_start:train_end],
        "val":   wide.loc[val_start:val_end],
        "test":  wide.loc[test_start:test_end],
    }
    for name, sp in splits_raw.items():
        print(f"  {name}: {sp.index[0]} ~ {sp.index[-1]} ({len(sp)} hours)")
    return splits_raw, wide.columns.tolist()


def _stack_split(wide_df: pd.DataFrame, stations: list, features: list = FEATURE_ORDER):
    """Wide table → (T, N, F) float32 array with the requested station order."""
    col_names = wide_df.columns.tolist()
    arrays = []
    for sta in stations:
        sta_cols = [f"{feat}_{sta}" for feat in features]
        if not all(c in col_names for c in sta_cols):
            print(f"  [warn] station {sta}: some feature columns missing, zero-filling")
        sta_df = pd.DataFrame(index=wide_df.index)
        for c in sta_cols:
            sta_df[c] = wide_df[c] if c in wide_df.columns else 0.0
        arrays.append(sta_df.values)
    return np.stack(arrays, axis=1).astype(np.float32)


def make_sliding_windows(stacked_norm: np.ndarray, seq_len: int, pred_len: int,
                         target_idx: int = 0):
    """Produce (samples, N, seq_len, F) X and (samples, N, pred_len) y arrays.

    Windows that span any NaN-holding timestep are dropped rather than having
    their holes filled — guarantees every training sample is fully observed.
    """
    T, N, F = stacked_norm.shape
    valid_mask = np.all(np.isfinite(stacked_norm), axis=(1, 2))
    total = T - seq_len - pred_len + 1

    X_list, y_list = [], []
    skipped = 0
    for i in range(total):
        if not np.all(valid_mask[i: i + seq_len + pred_len]):
            skipped += 1
            continue
        x = stacked_norm[i: i + seq_len]
        y = stacked_norm[i + seq_len: i + seq_len + pred_len, :, target_idx]
        X_list.append(x.transpose(1, 0, 2))
        y_list.append(y.T)
    if skipped:
        print(f"    dropped {skipped}/{total} windows ({skipped/total*100:.1f}%) "
              "due to NaNs")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def build_and_save_npz(splits_raw: dict, stations: list, output_dir: Path,
                       seq_len: int = SEQ_LEN, pred_lens: list = PRED_LENS):
    """Impute per-station, fit Min-Max scaler on TRAIN ONLY (prevents leakage),
    then emit the sliding-window NPZs + PM2.5 scaler for each pred_len.
    """
    target_feat = "PM2.5"
    target_idx  = FEATURE_ORDER.index(target_feat)

    print("\n  Imputing missing values...")
    stacked_splits = {}
    for split_name, wide_df in splits_raw.items():
        filled_cols = {}
        # Impute each station independently — no cross-station leakage.
        for sta in stations:
            sta_col_to_feat = {f"{feat}_{sta}": feat for feat in FEATURE_ORDER}
            avail = {c: f for c, f in sta_col_to_feat.items() if c in wide_df.columns}
            sta_df = wide_df[[c for c in avail]].copy()
            sta_df.columns = [avail[c] for c in sta_df.columns]
            sta_df_filled = impute_missing(sta_df)
            for feat in FEATURE_ORDER:
                filled_cols[f"{feat}_{sta}"] = (
                    sta_df_filled[feat] if feat in sta_df_filled.columns else np.nan
                )
        filled_wide = pd.DataFrame(filled_cols, index=wide_df.index)
        stacked_splits[split_name] = _stack_split(filled_wide, stations)

    train_stacked = stacked_splits["train"]
    _, _, F = train_stacked.shape

    print("\n  Fitting Min-Max scaler on training split only...")
    flat_train = train_stacked.reshape(-1, F)
    feat_min = np.nanmin(flat_train, axis=0)
    feat_max = np.nanmax(flat_train, axis=0)
    scale    = feat_max - feat_min
    scale[scale == 0] = 1.0
    target_scaler = np.array([feat_min[target_idx], feat_max[target_idx]], dtype=np.float32)
    print(f"    PM2.5 scaler: min={feat_min[target_idx]:.2f}, max={feat_max[target_idx]:.2f}")

    # Apply train-fit scaler to every split; leave NaNs NaN so the sliding
    # window step can reject any sample containing unobserved data.
    stacked_norm = {name: (arr - feat_min) / scale for name, arr in stacked_splits.items()}

    for pred_len in pred_lens:
        out_subdir = output_dir / "train_val_test_data" / f"{seq_len}_{pred_len}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n  pred_len={pred_len}:")
        for split_name in ["train", "val", "test"]:
            X, y = make_sliding_windows(stacked_norm[split_name], seq_len, pred_len, target_idx)
            path = out_subdir / f"{split_name}_PM25.npz"
            np.savez_compressed(path, X=X, y=y)
            print(f"    {split_name}: X={X.shape}, y={y.shape} → {path}")

        scaler_path = out_subdir / "scaler_PM25.npy"
        np.save(scaler_path, target_scaler)
        print(f"    scaler saved: {scaler_path}")

    return feat_min, feat_max


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: location.csv + AQI_processed for graph construction
# ──────────────────────────────────────────────────────────────────────────────
def save_location(stations: list, output_dir: Path) -> pd.DataFrame:
    """Pull lat/lon for each station from the national station list."""
    station_df = pd.read_csv(STATION_LIST_CSV, encoding="utf-8")
    target = station_df[station_df["监测点编码"].isin(stations)].copy()
    target = target.set_index("监测点编码").reindex(stations).reset_index()

    loc_df = pd.DataFrame({
        "site_name": target["监测点编码"].values,
        "Latitude":  target["纬度"].values,
        "Longitude": target["经度"].values,
    })
    loc_dir = output_dir / "location"
    loc_dir.mkdir(parents=True, exist_ok=True)
    out_path = loc_dir / "location.csv"
    loc_df.to_csv(out_path, index=False)
    print(f"  location.csv saved: {out_path}")
    return loc_df


def save_aqi_processed(merged: pd.DataFrame, stations: list, output_dir: Path,
                       train_start: str, train_end: str):
    """Save per-station AQI CSVs (training-window only) for S-graph construction.

    Restricting to the training window prevents future information from
    leaking into the Jensen-Shannon similarity graph.
    """
    aqi_dir = output_dir / "AQI_processed"
    aqi_dir.mkdir(parents=True, exist_ok=True)
    for i, station in enumerate(stations):
        try:
            st_df = merged.xs(station, level="station").loc[train_start:train_end]
        except KeyError:
            print(f"  [warn] station {station}: no data, skipping")
            continue
        st_df.to_csv(aqi_dir / f"PRSA_Data_{i+1}.csv")
    print(f"  AQI_processed saved ({len(stations)} stations, train window only)")


# ──────────────────────────────────────────────────────────────────────────────
# Subcommands
# ──────────────────────────────────────────────────────────────────────────────
def cmd_audit(args):
    cfg = CITY_CONFIGS[args.city]
    stations = cfg["stations"]
    start, end = cfg["train_start"], cfg["test_end"]
    out_dir = ROOT / "dataset" / cfg["dataset_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nData audit: {cfg['dataset_name']}  ({start} ~ {end})\n{'='*60}")

    print("\n[1/3] Loading AQ data...")
    aq_df = load_aq_data(stations, start, end)

    print("\n[2/3] Loading meteorology...")
    meteo_df = load_meteo_data(
        cfg["meteo_station"], start, end,
        cfg["meteo_altitude_m"], cfg["pres_strategy"],
        cfg.get("pres_climate_mean", 966.0),
        cfg.get("pres_climate_std", 8.0),
    )

    print("\n[3/3] Merging and auditing...")
    merged = build_merged_table(aq_df, meteo_df, stations, start, end)
    report = audit_coverage(merged, stations)

    pm25_report = report[report["feature"] == "PM2.5"]
    print("\n  PM2.5 coverage:")
    print(pm25_report[["station", "coverage_pct", "max_gap_hours"]].to_string(index=False))

    report_path = out_dir / "raw_coverage_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\n  full report saved: {report_path}")

    manifest = save_location(stations, out_dir)
    manifest.to_csv(out_dir / "station_manifest.csv", index=False)

    print("\nDone. If any station exceeds 24h max_gap_hours, consider excluding it.")


def cmd_preprocess(args):
    cfg = CITY_CONFIGS[args.city]
    stations = cfg["stations"]
    dataset_name = cfg["dataset_name"]
    out_dir = ROOT / "dataset" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = cfg["train_start"], cfg["test_end"]

    print(f"\n{'='*60}\nPreprocess: {dataset_name}")
    print(f"  train: {cfg['train_start']} ~ {cfg['train_end']}")
    print(f"  val:   {cfg['val_start']} ~ {cfg['val_end']}")
    print(f"  test:  {cfg['test_start']} ~ {cfg['test_end']}")
    print(f"  stations: {stations}\n{'='*60}")

    print("\n[Step 1] Loading AQ data...")
    aq_df = load_aq_data(stations, start, end)

    # Drop stations with (near-)zero PM2.5 coverage — sometimes stations in the
    # configured list are retired or renamed by the time the data covers them.
    MIN_COVERAGE = 5.0
    aq_stations_present = set(aq_df.index.get_level_values("station"))
    valid_stations = []
    for s in stations:
        if s not in aq_stations_present:
            print(f"  [warn] station {s}: missing entirely from AQ data, dropping")
            continue
        pm25_vals = aq_df.xs(s, level="station")["PM2.5"] if "PM2.5" in aq_df.columns else pd.Series()
        coverage = pm25_vals.notna().mean() * 100 if len(pm25_vals) > 0 else 0.0
        if coverage < MIN_COVERAGE:
            print(f"  [warn] station {s}: PM2.5 coverage {coverage:.1f}% < {MIN_COVERAGE}%, dropping")
        else:
            valid_stations.append(s)
    if len(valid_stations) < len(stations):
        print(f"  Using {len(valid_stations)}/{len(stations)} stations")
        stations = valid_stations

    print("\n[Step 2] Loading meteorology...")
    meteo_df = load_meteo_data(
        cfg["meteo_station"], start, end,
        cfg["meteo_altitude_m"], cfg["pres_strategy"],
        cfg.get("pres_climate_mean", 966.0),
        cfg.get("pres_climate_std", 8.0),
    )

    print("\n[Step 3] Building 12-feature panel...")
    merged = build_merged_table(aq_df, meteo_df, stations, start, end)

    print("\n[Step 4] Coverage audit...")
    report = audit_coverage(merged, stations)
    report.to_csv(out_dir / "raw_coverage_report.csv", index=False)
    pm25_mean = report[report["feature"] == "PM2.5"]["coverage_pct"].mean()
    print(f"  mean PM2.5 coverage: {pm25_mean:.1f}%")

    print("\n[Step 5] Time splits...")
    splits_raw, _ = build_splits(
        merged, stations,
        cfg["train_start"], cfg["train_end"],
        cfg["val_start"],   cfg["val_end"],
        cfg["test_start"],  cfg["test_end"],
    )

    split_manifest = {
        "dataset": dataset_name,
        "train": {"start": cfg["train_start"], "end": cfg["train_end"]},
        "val":   {"start": cfg["val_start"],   "end": cfg["val_end"]},
        "test":  {"start": cfg["test_start"],  "end": cfg["test_end"]},
        "stations": stations,
        "features": FEATURE_ORDER,
        "seq_len": SEQ_LEN,
        "pred_lens": PRED_LENS,
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)

    print("\n[Step 6] Writing NPZ files (scaler fit on train only)...")
    build_and_save_npz(splits_raw, stations, out_dir)

    print("\n[Step 7] Saving location.csv...")
    save_location(stations, out_dir)

    print("\n[Step 8] Saving AQI_processed (train window only) for S-graph...")
    save_aqi_processed(merged, stations, out_dir, cfg["train_start"], cfg["train_end"])

    print(f"\n{'='*60}\nDone. Output: {out_dir}")
    print(f"Next: python graphs/build_graphs.py --dataset {dataset_name}\n{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build DMGENet CN-city datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="Data-quality audit only.")
    p_audit.add_argument("--city", required=True, choices=list(CITY_CONFIGS.keys()))

    p_pre = sub.add_parser("preprocess", help="Full preprocessing pipeline.")
    p_pre.add_argument("--city", required=True, choices=list(CITY_CONFIGS.keys()))

    args = parser.parse_args()
    if args.cmd == "audit":
        cmd_audit(args)
    elif args.cmd == "preprocess":
        cmd_preprocess(args)


if __name__ == "__main__":
    main()
