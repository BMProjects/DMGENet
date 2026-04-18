"""
Download and preprocess Delhi multi-station air-quality data.

Source: CPCB (Central Pollution Control Board, India) via Kaggle —
"Air Quality Data in India (2015-2020)" by Rohan Rao. 12 Delhi NCT
stations, hourly cadence, PM2.5 + other pollutants.

Usage:
  1. Download the Kaggle dataset "rohanrao/air-quality-data-in-india":
       kaggle datasets download -d rohanrao/air-quality-data-in-india
  2. Unzip into dataset/Delhi_NCT/raw/
  3. Run: python data/download_delhi.py
"""

import glob
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset", "Delhi_NCT")
RAW_DIR     = os.path.join(DATASET_DIR, "raw")
AQI_DIR     = os.path.join(DATASET_DIR, "AQI_processed")

# Main Delhi NCT monitoring stations — counterpart to the Beijing 12-site panel.
DELHI_STATIONS = [
    "Anand Vihar, Delhi - DPCC",
    "Ashok Vihar, Delhi - DPCC",
    "Bawana, Delhi - DPCC",
    "CRRI Mathura Road, Delhi - IMD",
    "DTU, Delhi - CPCB",
    "Dwarka-Sector 8, Delhi - DPCC",
    "IGI Airport (T3), Delhi - IMD",
    "ITO, Delhi - CPCB",
    "Jahangirpuri, Delhi - DPCC",
    "Jawaharlal Nehru Stadium, Delhi - DPCC",
    "Lodhi Road, Delhi - IMD",
    "Mandir Marg, Delhi - DPCC",
    "Major Dhyan Chand National Stadium, Delhi - DPCC",
    "NSIT Dwarka, Delhi - CPCB",
    "North Campus, DU, Delhi - IMD",
    "Okhla Phase-2, Delhi - DPCC",
    "Patparganj, Delhi - DPCC",
    "Punjabi Bagh, Delhi - DPCC",
    "Pusa, Delhi - DPCC",
    "R K Puram, Delhi - DPCC",
    "Rohini, Delhi - DPCC",
    "Shadipur, Delhi - CPCB",
    "Siri Fort, Delhi - CPCB",
    "Sri Aurobindo Marg, Delhi - DPCC",
    "Vivek Vihar, Delhi - DPCC",
    "Wazirpur, Delhi - DPCC",
]

# Keep the 12 stations with highest PM2.5 coverage.
TARGET_N_STATIONS = 12

# Kaggle dataset ships pollutants only — no meteorology. To match the
# Beijing 12-dim layout, we use 6 pollutants + 6 extra trace-gas features.
POLLUTANTS     = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
EXTRA_FEATURES = ["NO", "NOx", "NH3", "Benzene", "Toluene", "Xylene"]
ALL_FEATURES   = POLLUTANTS + EXTRA_FEATURES  # 12 dims

TARGET          = "PM2.5"
TARGET_FILENAME = "PM25"


def download_kaggle_data():
    """Try kaggle CLI; fall back to manual-download instructions."""
    zip_path = os.path.join(RAW_DIR, "air-quality-data-in-india.zip")
    if os.path.exists(zip_path):
        print(f"Already present: {zip_path}")
        return zip_path

    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        print("Attempting kaggle CLI download...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "rohanrao/air-quality-data-in-india",
            "-p", RAW_DIR,
        ], check=True, capture_output=True, text=True)
        print(f"Downloaded: {zip_path}")
        return zip_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"kaggle CLI unavailable: {e}")
        print("\nPlease download manually:")
        print("  1. https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
        print("  2. Download the zip.")
        print(f"  3. Unzip the CSVs into {RAW_DIR}/")
        print("  4. Re-run this script.")

        csvs = glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if csvs:
            print(f"\nFound {len(csvs)} CSVs already in place, continuing...")
            return None
        sys.exit(1)


def extract_data():
    zip_path = os.path.join(RAW_DIR, "air-quality-data-in-india.zip")
    if os.path.exists(zip_path):
        print(f"Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)


def load_and_select_stations():
    """Load Delhi rows and pick the 12 stations with the most PM2.5 samples.

    Preferred layout: station_hour.csv (StationId, Datetime, pollutant cols)
    plus stations.csv (StationId, StationName, City). Falls back to older
    single-CSV layouts that have a `city` column.
    """
    hour_path     = os.path.join(RAW_DIR, "station_hour.csv")
    stations_path = os.path.join(RAW_DIR, "stations.csv")

    if not os.path.exists(hour_path):
        csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
        print(f"station_hour.csv missing; trying {len(csv_files)} other CSVs...")
        all_data = None
        for f in csv_files:
            df = pd.read_csv(f, low_memory=False)
            cols_lower = [c.lower() for c in df.columns]
            if "city" in cols_lower:
                all_data = df if all_data is None else pd.concat([all_data, df], ignore_index=True)
        if all_data is None:
            print("Unrecognised data layout; check dataset/Delhi_NCT/raw/")
            sys.exit(1)
        all_data.rename(
            columns={c: "datetime" for c in all_data.columns if c.lower() == "date"},
            inplace=True,
        )
        all_data.rename(
            columns={c: "city" for c in all_data.columns if c.lower() == "city"},
            inplace=True,
        )
        delhi_data = all_data[all_data["city"].str.contains("Delhi", case=False, na=False)].copy()
        delhi_data.rename(
            columns={c: "station" for c in delhi_data.columns
                     if "station" in c.lower() or "site" in c.lower()},
            inplace=True,
        )
    else:
        print(f"Loading {hour_path}...")
        all_data = pd.read_csv(hour_path, low_memory=False)
        print(f"  columns: {list(all_data.columns)}")
        print(f"  rows: {len(all_data):,}")

        all_data.rename(columns={"StationId": "station", "Datetime": "datetime"}, inplace=True)

        if os.path.exists(stations_path):
            meta = pd.read_csv(stations_path)
            meta.rename(columns={"StationId": "station"}, inplace=True)
            all_data = all_data.merge(
                meta[["station", "StationName", "City"]], on="station", how="left"
            )
            delhi_mask = all_data["City"].str.contains("Delhi", case=False, na=False)
        else:
            # Delhi station IDs all start with "DL".
            delhi_mask = all_data["station"].str.startswith("DL", na=False)

        delhi_data = all_data[delhi_mask].copy()

    print(f"Delhi rows: {len(delhi_data):,}")
    stations = delhi_data["station"].unique()
    print(f"Delhi stations: {len(stations)}")

    pm25_col = "PM2.5" if "PM2.5" in delhi_data.columns else TARGET
    station_coverage = {
        st: delhi_data.loc[delhi_data["station"] == st, pm25_col].notna().sum()
        for st in stations
    }

    sorted_stations = sorted(station_coverage.items(), key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in sorted_stations[:TARGET_N_STATIONS]]

    print(f"\nSelected {TARGET_N_STATIONS} stations (by PM2.5 coverage):")
    for i, (st, cov) in enumerate(sorted_stations[:TARGET_N_STATIONS]):
        name = ""
        if "StationName" in delhi_data.columns:
            nm = delhi_data.loc[delhi_data["station"] == st, "StationName"].dropna()
            name = f" ({nm.iloc[0]})" if len(nm) > 0 else ""
        print(f"  {i+1:2d}. {st}{name}  PM2.5 records: {cov:,}")

    return delhi_data, selected


def process_and_align(delhi_data, selected_stations):
    """Align all stations to a common hourly index and interpolate gaps."""
    if "datetime" in delhi_data.columns:
        delhi_data["datetime"] = pd.to_datetime(delhi_data["datetime"], errors="coerce")
    elif "Date" in delhi_data.columns:
        delhi_data["datetime"] = pd.to_datetime(delhi_data["Date"], errors="coerce")

    delhi_data = delhi_data.dropna(subset=["datetime"]).set_index("datetime")

    time_ranges = []
    for st in selected_stations:
        st_data = delhi_data[delhi_data["station"] == st]
        if len(st_data) > 0:
            time_ranges.append((st_data.index.min(), st_data.index.max()))

    common_start = max(tr[0] for tr in time_ranges)
    common_end   = min(tr[1] for tr in time_ranges)
    print(f"\nCommon span: {common_start} ~ {common_end}")

    time_index = pd.date_range(start=common_start, end=common_end, freq="h")
    print(f"Total timesteps: {len(time_index)}")

    all_station_data = {}
    os.makedirs(AQI_DIR, exist_ok=True)

    for st in selected_stations:
        st_data = delhi_data[delhi_data["station"] == st].copy()
        st_data = st_data[~st_data.index.duplicated(keep="first")]

        for feat in ALL_FEATURES:
            if feat not in st_data.columns:
                st_data[feat] = np.nan

        st_data = st_data[ALL_FEATURES].reindex(time_index)

        for col in ALL_FEATURES:
            st_data[col] = pd.to_numeric(st_data[col], errors="coerce")

        # Linear interpolation for short gaps, ffill/bfill for edges,
        # column mean for anything still missing. A column that is fully
        # NaN falls back to 0 so downstream normalisation stays finite.
        st_data = st_data.interpolate(method="linear", limit=24)
        st_data = st_data.ffill(limit=6).bfill(limit=6)
        for col in ALL_FEATURES:
            if st_data[col].isna().any():
                st_data[col].fillna(st_data[col].mean(), inplace=True)
            if st_data[col].isna().all():
                st_data[col].fillna(0, inplace=True)

        coverage = st_data.notna().mean().mean() * 100
        safe_name = st.replace(",", "").replace(" ", "_").replace("-", "_")[:30]
        all_station_data[safe_name] = st_data.values

        csv_path = os.path.join(AQI_DIR, f"{safe_name}.csv")
        st_data.to_csv(csv_path)
        print(f"  {safe_name}: shape={st_data.shape}, coverage={coverage:.1f}%")

    return all_station_data, time_index, ALL_FEATURES


def create_sliding_window(all_station_data, feature_cols, seq_len, pred_len, target=TARGET):
    """Build (samples, N, seq_len, F) X and (samples, N, pred_len) y arrays."""
    station_names = list(all_station_data.keys())
    N = len(station_names)
    T = len(list(all_station_data.values())[0])
    F = len(feature_cols)

    data_array = np.stack([all_station_data[st] for st in station_names], axis=1)
    assert data_array.shape == (T, N, F), f"Shape mismatch: {data_array.shape} vs ({T},{N},{F})"

    target_idx = feature_cols.index(target) if target in feature_cols else 0
    target_series = data_array[:, :, target_idx]
    _min = np.nanmin(target_series)
    _max = np.nanmax(target_series)

    data_normed = np.zeros_like(data_array)
    for f_idx in range(F):
        f_min = np.nanmin(data_array[:, :, f_idx])
        f_max = np.nanmax(data_array[:, :, f_idx])
        if f_max - f_min > 1e-8:
            data_normed[:, :, f_idx] = (data_array[:, :, f_idx] - f_min) / (f_max - f_min)

    num_samples = T - seq_len - pred_len + 1
    X = np.zeros((num_samples, N, seq_len, F), dtype=np.float32)
    y = np.zeros((num_samples, N, pred_len), dtype=np.float32)

    for i in range(num_samples):
        X[i] = data_normed[i: i + seq_len].transpose(1, 0, 2)
        y[i] = data_normed[i + seq_len: i + seq_len + pred_len, :, target_idx].T

    n_train = int(num_samples * 0.6)
    n_val   = int(num_samples * 0.2)
    splits = {
        "train": (X[:n_train],                           y[:n_train]),
        "val":   (X[n_train: n_train + n_val],           y[n_train: n_train + n_val]),
        "test":  (X[n_train + n_val:],                   y[n_train + n_val:]),
    }

    print(f"  windows: X=({num_samples},{N},{seq_len},{F}), y=({num_samples},{N},{pred_len})")
    for name, (sx, sy) in splits.items():
        print(f"    {name}: X={sx.shape}, y={sy.shape}")

    return splits, np.array([_min, _max])


def save_splits(splits, scaler, seq_len, pred_len, target=TARGET_FILENAME):
    out_dir = os.path.join(DATASET_DIR, "train_val_test_data", f"{seq_len}_{pred_len}")
    os.makedirs(out_dir, exist_ok=True)

    for name, (X, y) in splits.items():
        path = os.path.join(out_dir, f"{name}_{target}.npz")
        np.savez_compressed(path, X=X, y=y)
        print(f"  saved: {path}")

    scaler_path = os.path.join(out_dir, f"scaler_{target}.npy")
    np.save(scaler_path, scaler)
    print(f"  saved: {scaler_path} (min={scaler[0]:.4f}, max={scaler[1]:.4f})")


def save_station_metadata(selected_stations, all_station_data):
    meta = {
        "stations": list(all_station_data.keys()),
        "n_stations": len(all_station_data),
        "features": ALL_FEATURES,
        "original_names": selected_stations[:len(all_station_data)],
    }
    meta_path = os.path.join(DATASET_DIR, "station_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Station metadata: {meta_path}")


def main():
    print("=" * 60)
    print("Delhi NCT multi-station air-quality data preparation")
    print("=" * 60)

    print("\n--- Step 1: download ---")
    download_kaggle_data()
    extract_data()

    print("\n--- Step 2: load + select stations ---")
    delhi_data, selected_stations = load_and_select_stations()

    print("\n--- Step 3: align + clean ---")
    all_station_data, time_index, feature_cols = process_and_align(delhi_data, selected_stations)

    save_station_metadata(selected_stations, all_station_data)

    print("\n--- Step 4: build sliding-window splits ---")
    for pred_len in [1, 6, 12, 24]:
        print(f"\n  --- seq_len=72, pred_len={pred_len} ---")
        splits, scaler = create_sliding_window(all_station_data, feature_cols, 72, pred_len)
        save_splits(splits, scaler, 72, pred_len)

    print("\n" + "=" * 60)
    print("Delhi data ready.")
    print(f"  AQI panels:      {AQI_DIR}")
    print(f"  Training splits: {os.path.join(DATASET_DIR, 'train_val_test_data')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
