"""
Download and preprocess the UCI Beijing Multi-Site Air Quality dataset.

Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
12 monitoring stations, hourly air-quality + meteorology, 2013-03 — 2017-02.

Outputs:
  dataset/Beijing_12/AQI_processed/PRSA_Data_{1..12}.csv
      — per-station hourly panel (used by graph builders)
  dataset/Beijing_12/train_val_test_data/72_{pred}/train_PM25.npz (+ val/test)
      — sliding-window samples keyed (samples, N, seq_len, features)

Usage:
  python data/download_beijing.py
"""

import os
import sys
import zipfile

import numpy as np


def require(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"Missing dependency: {pkg}. Run: pip install {pkg}")
        sys.exit(1)


pd = require("pandas")
requests = require("requests")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(ROOT, "dataset", "Beijing_12", "raw")
AQI_DIR  = os.path.join(ROOT, "dataset", "Beijing_12", "AQI_processed")
DATA_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

# Station order must match dataset/Beijing_12/location/location.csv.
STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi",
    "Guanyuan",     "Gucheng",   "Huairou",  "Nongzhanguan",
    "Shunyi",       "Tiantan",   "Wanliu",   "Wanshouxigong",
]

# Feature layout (11 raw + 1 derived wind-direction = 12 dims, same as the paper).
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO      = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
# Raw data has no numeric wind-direction column, only `wd` (compass string).
# It gets encoded to WD_deg below to fill out the 12-dim feature vector.

TARGET          = "PM2.5"   # column name in the raw CSVs
TARGET_FILENAME = "PM25"    # file name convention used by training scripts


def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)

    zip_path = os.path.join(RAW_DIR, "beijing_air_quality.zip")
    if os.path.exists(zip_path):
        print(f"Already present: {zip_path}, skipping download")
    else:
        print(f"Downloading: {DATA_URL}")
        resp = requests.get(DATA_URL, timeout=120)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        print(f"Downloaded: {zip_path} ({len(resp.content)/1024/1024:.1f} MB)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted: {RAW_DIR}")

    # UCI ships a zip containing a second zip; unwrap one more level.
    inner_zips = [f for f in os.listdir(RAW_DIR)
                  if f.endswith(".zip") and f != os.path.basename(zip_path)]
    for iz in inner_zips:
        with zipfile.ZipFile(os.path.join(RAW_DIR, iz), "r") as zf2:
            zf2.extractall(RAW_DIR)
        print(f"Extracted inner: {iz}")

    csvs = []
    for dirpath, _, filenames in os.walk(RAW_DIR):
        for fn in filenames:
            if fn.startswith("PRSA_Data_") and fn.endswith(".csv"):
                csvs.append(os.path.join(dirpath, fn))
    print(f"Found {len(csvs)} station CSVs")
    for c in sorted(csvs):
        print(f"  {c}")
    return csvs


def load_and_clean(csv_files):
    """Read the 12 station CSVs, align on a common time index, interpolate gaps."""
    station_dfs = {}

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "station" in df.columns:
            stn = df["station"].iloc[0]
        else:
            fn = os.path.basename(csv_path)
            stn = fn.replace("PRSA_Data_", "").replace("_20130301-20170228.csv", "").strip()

        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.set_index("datetime").sort_index()

        # Encode the 16-point compass `wd` string to degrees.
        if "wd" in df.columns:
            wind_dir_map = {
                "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
                "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
                "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
                "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
            }
            df["WD_deg"] = df["wd"].map(wind_dir_map)

        station_dfs[stn] = df

    found = sorted(station_dfs.keys())
    print(f"Loaded {len(found)} stations: {found}")

    common_start = max(df.index.min() for df in station_dfs.values())
    common_end   = min(df.index.max() for df in station_dfs.values())
    time_idx = pd.date_range(common_start, common_end, freq="h")
    print(f"Common span: {common_start} ~ {common_end} ({len(time_idx)} hours)")

    first_df = list(station_dfs.values())[0]
    feature_cols = POLLUTANTS + METEO + (["WD_deg"] if "WD_deg" in first_df.columns else [])

    all_data = {}
    for stn in STATIONS:
        # Tolerate whitespace / underscore differences in source station names.
        matched = None
        for k in station_dfs:
            if k.lower().replace(" ", "").replace("_", "") == stn.lower():
                matched = k
                break
        if matched is None:
            for k in station_dfs:
                if stn.lower()[:5] in k.lower():
                    matched = k
                    break
        if matched is None:
            print(f"[warn] station not matched: {stn}")
            continue

        df = station_dfs[matched].reindex(time_idx)
        # Paper §5.2: linear interpolation for missing hourly values.
        df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
        df[feature_cols] = df[feature_cols].ffill().bfill()
        all_data[stn] = df[feature_cols]

    print(f"Processed {len(all_data)} stations")
    return all_data, time_idx, feature_cols


def save_aqi_processed(all_data):
    """Save per-station panels as PRSA_Data_{i}.csv for graph construction."""
    os.makedirs(AQI_DIR, exist_ok=True)
    for i, stn in enumerate(STATIONS):
        if stn not in all_data:
            continue
        out_path = os.path.join(AQI_DIR, f"PRSA_Data_{i+1}.csv")
        all_data[stn].to_csv(out_path)
        print(f"  saved: {out_path} (shape={all_data[stn].shape})")


def create_sliding_window_dataset(all_data, feature_cols, seq_len, pred_len, target=TARGET):
    """Build the multi-station sliding-window arrays.

    X: (samples, N, seq_len, features)
    y: (samples, N, pred_len)  — target variable only
    """
    target_idx = feature_cols.index(target) if target in feature_cols else 0

    arrays = []
    for stn in STATIONS:
        if stn in all_data:
            arrays.append(all_data[stn].values)
        else:
            print(f"[warn] station {stn} missing, zero-filling")
            T = len(list(all_data.values())[0])
            arrays.append(np.zeros((T, len(feature_cols))))

    stacked = np.stack(arrays, axis=1).astype(np.float32)   # (T, N, F)
    T, N, F = stacked.shape
    print(f"Stacked shape: T={T}, N={N}, F={F}")

    # Global Min-Max over all stations per feature; matches CityDataLoader's
    # inverse_transform which reads a single [min, max] per target.
    feat_min = stacked.reshape(-1, F).min(axis=0)
    feat_max = stacked.reshape(-1, F).max(axis=0)
    scale = feat_max - feat_min
    scale[scale == 0] = 1.0
    stacked_norm = (stacked - feat_min) / scale

    X_list, y_list = [], []
    total_len = T - seq_len - pred_len + 1
    for i in range(total_len):
        x = stacked_norm[i: i + seq_len]
        y = stacked_norm[i + seq_len: i + seq_len + pred_len, :, target_idx]
        X_list.append(x.transpose(1, 0, 2))   # (N, seq_len, F)
        y_list.append(y.T)                    # (N, pred_len)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Windows: X={X.shape}, y={y.shape}")

    # Paper §5.1: 6/2/2 chronological split.
    n = len(X)
    n_train = int(n * 0.6)
    n_val   = int(n * 0.2)
    splits = {
        "train": (X[:n_train],                           y[:n_train]),
        "val":   (X[n_train: n_train + n_val],           y[n_train: n_train + n_val]),
        "test":  (X[n_train + n_val:],                   y[n_train + n_val:]),
    }
    for name, (sx, sy) in splits.items():
        print(f"  {name}: X={sx.shape}, y={sy.shape}")

    target_scaler = np.array([feat_min[target_idx], feat_max[target_idx]], dtype=np.float32)
    return splits, target_scaler


def save_splits(splits, scaler, seq_len, pred_len, target=TARGET_FILENAME):
    """Save NPZ splits matching the CityDataLoader naming convention."""
    out_dir = os.path.join(ROOT, "dataset", "Beijing_12", "train_val_test_data",
                           f"{seq_len}_{pred_len}")
    os.makedirs(out_dir, exist_ok=True)

    for name, (X, y) in splits.items():
        path = os.path.join(out_dir, f"{name}_{target}.npz")
        np.savez_compressed(path, X=X, y=y)
        print(f"  saved: {path}")

    scaler_path = os.path.join(out_dir, f"scaler_{target}.npy")
    np.save(scaler_path, scaler)
    print(f"  saved: {scaler_path} (min={scaler[0]:.4f}, max={scaler[1]:.4f})")


def main():
    print("=" * 60 + "\n[1/4] Download\n" + "=" * 60)
    csv_files = download_data()

    print("\n" + "=" * 60 + "\n[2/4] Load and clean\n" + "=" * 60)
    all_data, time_idx, feature_cols = load_and_clean(csv_files)
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Total hours: {len(time_idx)}")

    print("\n" + "=" * 60 + "\n[3/4] Save AQI_processed (for graph construction)\n" + "=" * 60)
    save_aqi_processed(all_data)

    print("\n" + "=" * 60 + "\n[4/4] Build sliding-window splits\n" + "=" * 60)
    # The trained models use seq_len=72 hours (not 24 as the paper claims).
    # This discrepancy is preserved to match the reported experimental results.
    seq_len = 72
    for pred_len in [1, 3, 6, 12, 24]:
        print(f"\n--- seq_len={seq_len}, pred_len={pred_len} ---")
        splits, scaler = create_sliding_window_dataset(
            all_data, feature_cols, seq_len, pred_len
        )
        save_splits(splits, scaler, seq_len, pred_len)

    print("\nDone.")
    print(f"  AQI panels:       {AQI_DIR}")
    print(f"  Training splits:  {os.path.join(ROOT, 'dataset', 'Beijing_12', 'train_val_test_data')}")


if __name__ == "__main__":
    main()
