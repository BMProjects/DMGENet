"""
Delhi_NCT dataset build — pollutants + NOAA-ISD meteorology.

Purpose:
  1. Join the existing Delhi_NCT air-quality stations with hourly NOAA
     ISD-Lite observations from the nearest official weather station.
  2. Produce the 12-feature layout used elsewhere in the project:
       [PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, WD_deg]
  3. Write to a mirrored dataset directory (default Delhi_NCT_Meteo) so
     the weather-augmented split does not overwrite the pollutant-only one.

Inputs:
  Air quality:  dataset/Delhi_NCT/raw/station_hour.csv
  Coordinates:  dataset/Delhi_NCT/location/location.csv
  Weather:      NOAA ISD-Lite
      station list:  https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
      hourly files:  https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{USAF}-{WBAN}-{year}.gz

Notes:
  - The legacy Delhi loader padded the 12-dim vector with extra trace gases
    (NO, NOx, NH3, Benzene, Toluene, Xylene). This script instead uses
    6 pollutants + 6 meteorological variables — the same layout as Beijing,
    which makes cross-city transfer/comparison meaningful.
  - PRES is mapped directly from ISD-Lite SLP (sea-level pressure); Delhi's
    elevation (~216 m) makes the correction term small enough to ignore.
  - All timestamps are converted to Asia/Kolkata and aligned to Delhi's raw
    AQ hourly index.

Usage:
  python data/build_delhi_dataset.py
  python data/build_delhi_dataset.py --output-dataset Delhi_NCT_Meteo
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATASET = ROOT / "dataset" / "Delhi_NCT"

AQ_FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO_FEATURES = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_deg"]
FEATURE_ORDER = AQ_FEATURES + METEO_FEATURES
PRED_LENS = [1, 6, 12, 24]
SEQ_LEN = 72
TIMEZONE = "Asia/Kolkata"
NOAA_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
NOAA_LITE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station_id}-{year}.gz"


@dataclass
class WeatherStation:
    station_id: str
    name: str
    lat: float
    lon: float
    elev_m: float
    begin: pd.Timestamp
    end: pd.Timestamp


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "-o", str(out_path), url], check=True)
        return
    with urllib.request.urlopen(url, timeout=120) as resp, out_path.open("wb") as f:
        shutil.copyfileobj(resp, f)


def load_location_frame() -> pd.DataFrame:
    path = SOURCE_DATASET / "location" / "location.csv"
    df = pd.read_csv(path)
    df["station_id"] = df["station_id"].astype(str)
    return df


def load_raw_delhi_hourly() -> pd.DataFrame:
    hour_path = SOURCE_DATASET / "raw" / "station_hour.csv"
    stations_path = SOURCE_DATASET / "raw" / "stations.csv"
    if not hour_path.exists():
        raise FileNotFoundError(f"Missing raw file: {hour_path}")
    df = pd.read_csv(hour_path, low_memory=False)
    df = df.rename(columns={"StationId": "station_id", "Datetime": "datetime"})
    if stations_path.exists():
        meta = pd.read_csv(stations_path).rename(columns={"StationId": "station_id"})
        use_cols = [c for c in ["station_id", "StationName", "City"] if c in meta.columns]
        if use_cols:
            df = df.merge(meta[use_cols], on="station_id", how="left")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df


def infer_common_time_range(df: pd.DataFrame, selected_station_ids: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    ranges = []
    for sid in selected_station_ids:
        st = df[df["station_id"] == sid].copy()
        if st.empty:
            raise ValueError(f"Delhi raw data missing station {sid}")
        ranges.append((st["datetime"].min(), st["datetime"].max()))
    common_start = max(x[0] for x in ranges)
    common_end = min(x[1] for x in ranges)
    return common_start.floor("h"), common_end.floor("h")


def load_noaa_history(cache_path: Path) -> list[WeatherStation]:
    download_file(NOAA_HISTORY_URL, cache_path)
    stations: list[WeatherStation] = []
    with cache_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
                elev = float(row["ELEV(M)"])
            except Exception:
                continue
            if row["CTRY"] != "IN":
                continue
            if not (27.0 <= lat <= 30.0 and 76.0 <= lon <= 79.0):
                continue
            station_id = f'{row["USAF"]}-{row["WBAN"]}'
            begin = pd.to_datetime(row["BEGIN"], format="%Y%m%d", errors="coerce")
            end = pd.to_datetime(row["END"], format="%Y%m%d", errors="coerce")
            if pd.isna(begin) or pd.isna(end):
                continue
            stations.append(
                WeatherStation(
                    station_id=station_id,
                    name=row["STATION NAME"].strip(),
                    lat=lat,
                    lon=lon,
                    elev_m=elev,
                    begin=begin,
                    end=end,
                )
            )
    return stations


def parse_isd_lite_bytes(content: bytes) -> pd.DataFrame:
    records = []
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        text = gz.read().decode("utf-8", errors="ignore").splitlines()
    for line in text:
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            year, mo, da, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        except Exception:
            continue

        def _fv(idx: int):
            if idx >= len(parts):
                return np.nan
            try:
                v = int(parts[idx])
            except Exception:
                return np.nan
            return np.nan if v == -9999 else v

        temp = _fv(4)
        dewp = _fv(5)
        slp = _fv(6)
        wdir = _fv(7)
        wspd = _fv(8)
        if len(parts) > 10:
            try:
                raw_p = int(parts[10])
            except Exception:
                raw_p = -9999
            if raw_p == -9999:
                prec = 0.0
            elif raw_p == -1:
                prec = 0.1
            else:
                prec = raw_p / 10.0
        else:
            prec = 0.0
        records.append(
            {
                "datetime_utc": pd.Timestamp(year=year, month=mo, day=da, hour=hr, tz="UTC"),
                "TEMP": temp / 10.0 if np.isfinite(temp) else np.nan,
                "DEWP": dewp / 10.0 if np.isfinite(dewp) else np.nan,
                "PRES": slp / 10.0 if np.isfinite(slp) else np.nan,  # use SLP as pressure feature
                "WD_deg": float(wdir) if np.isfinite(wdir) else np.nan,
                "WSPM": wspd / 10.0 if np.isfinite(wspd) else np.nan,
                "RAIN": float(prec),
            }
        )
    if not records:
        return pd.DataFrame(columns=["TEMP", "DEWP", "PRES", "WD_deg", "WSPM", "RAIN"])
    df = pd.DataFrame(records).set_index("datetime_utc").sort_index()
    df.index = df.index.tz_convert(TIMEZONE).tz_localize(None).round("h")
    df = df[~df.index.duplicated(keep="first")]
    return df


def download_isd_lite_station_year(station_id: str, year: int, cache_dir: Path) -> Path | None:
    out_path = cache_dir / f"{station_id}-{year}.gz"
    if out_path.exists():
        try:
            head = out_path.read_bytes()[:2]
        except Exception:
            head = b""
        if head == b"\x1f\x8b":
            return out_path
        out_path.unlink(missing_ok=True)
    url = NOAA_LITE_URL.format(year=year, station_id=station_id)
    try:
        download_file(url, out_path)
        head = out_path.read_bytes()[:2]
        if head != b"\x1f\x8b":
            out_path.unlink(missing_ok=True)
            return None
        return out_path
    except Exception:
        out_path.unlink(missing_ok=True)
        return None


def load_weather_station_timeseries(station_id: str, years: list[int], cache_dir: Path) -> pd.DataFrame:
    frames = []
    for year in years:
        fpath = download_isd_lite_station_year(station_id, year, cache_dir)
        if fpath is None or not fpath.exists():
            continue
        frames.append(parse_isd_lite_bytes(fpath.read_bytes()))
    if not frames:
        return pd.DataFrame(columns=METEO_FEATURES)
    df = pd.concat(frames).sort_index()
    return df[METEO_FEATURES]


def build_station_weather_mapping(
    aq_location: pd.DataFrame,
    weather_stations: list[WeatherStation],
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
    cache_dir: Path,
) -> tuple[dict[str, str], dict[str, pd.DataFrame], list[dict]]:
    years = list(range(common_start.year, common_end.year + 1))
    eligible = []
    for ws in weather_stations:
        if ws.begin <= common_start and ws.end >= common_end:
            eligible.append(ws)
    if not eligible:
        raise ValueError("No NOAA weather station covers the Delhi common time range.")

    station_cache: dict[str, pd.DataFrame] = {}
    chosen: dict[str, str] = {}
    audit_rows: list[dict] = []
    for _, row in aq_location.iterrows():
        sid = str(row["station_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        ranked = sorted(
            eligible,
            key=lambda ws: haversine_km(lat, lon, ws.lat, ws.lon),
        )
        selected = None
        for ws in ranked[:8]:
            if ws.station_id not in station_cache:
                station_cache[ws.station_id] = load_weather_station_timeseries(ws.station_id, years, cache_dir)
            wdf = station_cache[ws.station_id]
            if wdf.empty:
                continue
            cov = wdf.loc[common_start:common_end].reindex(
                pd.date_range(common_start, common_end, freq="h")
            )["TEMP"].notna().mean()
            if cov >= 0.70:
                selected = (ws, cov)
                break
        if selected is None:
            ws = ranked[0]
            if ws.station_id not in station_cache:
                station_cache[ws.station_id] = load_weather_station_timeseries(ws.station_id, years, cache_dir)
            cov = station_cache[ws.station_id].loc[common_start:common_end].reindex(
                pd.date_range(common_start, common_end, freq="h")
            )["TEMP"].notna().mean()
            selected = (ws, cov)
        ws, cov = selected
        chosen[sid] = ws.station_id
        audit_rows.append(
            {
                "station_id": sid,
                "aq_station_name": row["station_name"],
                "aq_lat": lat,
                "aq_lon": lon,
                "weather_station_id": ws.station_id,
                "weather_station_name": ws.name,
                "weather_lat": ws.lat,
                "weather_lon": ws.lon,
                "distance_km": haversine_km(lat, lon, ws.lat, ws.lon),
                "temp_coverage": cov,
            }
        )
    return chosen, station_cache, audit_rows


def fill_weather_frame(df: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = df.reindex(time_index).copy()
    for col in METEO_FEATURES:
        series = pd.to_numeric(out[col], errors="coerce")
        series = series.interpolate(method="time", limit=24, limit_direction="both")
        series = series.ffill(limit=6).bfill(limit=6)
        if series.isna().any():
            mean_val = float(series.mean()) if np.isfinite(series.mean()) else 0.0
            series = series.fillna(mean_val)
        out[col] = series.astype(np.float32)
    return out


def fill_aq_frame(df: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = df.reindex(time_index).copy()
    for col in AQ_FEATURES:
        series = pd.to_numeric(out[col], errors="coerce")
        series = series.interpolate(method="time", limit=24, limit_direction="both")
        series = series.ffill(limit=6).bfill(limit=6)
        if series.isna().any():
            mean_val = float(series.mean()) if np.isfinite(series.mean()) else 0.0
            series = series.fillna(mean_val)
        out[col] = series.astype(np.float32)
    return out


def create_sliding_window(all_station_data: dict[str, np.ndarray], feature_cols: list[str], seq_len: int, pred_len: int):
    station_names = list(all_station_data.keys())
    N = len(station_names)
    T = len(next(iter(all_station_data.values())))
    F = len(feature_cols)
    data_array = np.stack([all_station_data[st] for st in station_names], axis=1)  # (T,N,F)

    target_idx = feature_cols.index("PM2.5")
    target_series = data_array[:, :, target_idx]
    _min = np.nanmin(target_series)
    _max = np.nanmax(target_series)

    data_normed = np.zeros_like(data_array, dtype=np.float32)
    for f_idx in range(F):
        f_min = np.nanmin(data_array[:, :, f_idx])
        f_max = np.nanmax(data_array[:, :, f_idx])
        if f_max - f_min > 1e-8:
            data_normed[:, :, f_idx] = (data_array[:, :, f_idx] - f_min) / (f_max - f_min)
        else:
            data_normed[:, :, f_idx] = 0.0

    num_samples = T - seq_len - pred_len + 1
    X = np.zeros((num_samples, N, seq_len, F), dtype=np.float32)
    y = np.zeros((num_samples, N, pred_len), dtype=np.float32)
    for i in range(num_samples):
        X[i] = data_normed[i:i + seq_len].transpose(1, 0, 2)
        y[i] = data_normed[i + seq_len:i + seq_len + pred_len, :, target_idx].T

    n_train = int(num_samples * 0.6)
    n_val = int(num_samples * 0.2)
    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
    }
    scaler = np.array([_min, _max], dtype=np.float32)
    return splits, scaler


def save_splits(dataset_root: Path, splits: dict, scaler: np.ndarray, seq_len: int, pred_len: int):
    out_dir = dataset_root / "train_val_test_data" / f"{seq_len}_{pred_len}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (X, y) in splits.items():
        np.savez_compressed(out_dir / f"{name}_PM25.npz", X=X, y=y)
    np.save(out_dir / "scaler_PM25.npy", scaler)


def main():
    ap = argparse.ArgumentParser(description="Build NOAA-ISD weather-augmented Delhi dataset")
    ap.add_argument("--output-dataset", default="Delhi_NCT_Meteo")
    args = ap.parse_args()

    out_root = ROOT / "dataset" / args.output_dataset
    aqi_dir = out_root / "AQI_processed"
    loc_dir = out_root / "location"
    raw_weather_dir = out_root / "raw_weather" / "noaa_isd_lite"
    aqi_dir.mkdir(parents=True, exist_ok=True)
    loc_dir.mkdir(parents=True, exist_ok=True)
    raw_weather_dir.mkdir(parents=True, exist_ok=True)

    aq_location = load_location_frame()
    selected_station_ids = aq_location["station_id"].astype(str).tolist()
    shutil.copy2(SOURCE_DATASET / "location" / "location.csv", loc_dir / "location.csv")

    aq_raw = load_raw_delhi_hourly()
    common_start, common_end = infer_common_time_range(aq_raw, selected_station_ids)
    time_index = pd.date_range(common_start, common_end, freq="h")

    noaa_history = load_noaa_history(raw_weather_dir / "isd-history.csv")
    mapping, weather_station_cache, audit_rows = build_station_weather_mapping(
        aq_location=aq_location,
        weather_stations=noaa_history,
        common_start=common_start,
        common_end=common_end,
        cache_dir=raw_weather_dir,
    )

    # persist station-level weather source mapping
    pd.DataFrame(audit_rows).to_csv(out_root / "weather_station_mapping.csv", index=False)

    all_station_data: dict[str, np.ndarray] = {}
    station_meta = []
    for _, row in aq_location.iterrows():
        sid = str(row["station_id"])
        sname = str(row["station_name"])
        aq_df = aq_raw[aq_raw["station_id"] == sid].copy()
        aq_df = aq_df.set_index("datetime").sort_index()
        for col in AQ_FEATURES:
            if col not in aq_df.columns:
                aq_df[col] = np.nan
        aq_frame = fill_aq_frame(aq_df[AQ_FEATURES], time_index)

        wsid = mapping[sid]
        weather_df = weather_station_cache[wsid]
        weather_frame = fill_weather_frame(weather_df, time_index)

        merged = pd.concat([aq_frame, weather_frame], axis=1)[FEATURE_ORDER]
        merged.to_csv(aqi_dir / f"{sid}.csv")
        all_station_data[sid] = merged.to_numpy(dtype=np.float32)

        station_meta.append(
            {
                "station_id": sid,
                "station_name": sname,
                "weather_station_id": wsid,
                "weather_station_name": next((x["weather_station_name"] for x in audit_rows if x["station_id"] == sid), ""),
                "feature_order": FEATURE_ORDER,
            }
        )

    for pred_len in PRED_LENS:
        splits, scaler = create_sliding_window(all_station_data, FEATURE_ORDER, SEQ_LEN, pred_len)
        save_splits(out_root, splits, scaler, SEQ_LEN, pred_len)

    metadata = {
        "source_dataset": "Delhi_NCT",
        "weather_source": "NOAA ISD-Lite",
        "weather_history_url": NOAA_HISTORY_URL,
        "station_time_zone": TIMEZONE,
        "common_start": str(common_start),
        "common_end": str(common_end),
        "n_stations": len(selected_station_ids),
        "feature_order": FEATURE_ORDER,
        "pressure_note": "PRES is mapped from NOAA ISD-Lite SLP (sea-level pressure).",
        "split_mode": "legacy_6_2_2_window_split",
        "stations": station_meta,
    }
    (out_root / "station_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "weather_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("Delhi NOAA weather-augmented dataset built successfully")
    print(f"Output: {out_root}")
    print(f"Common time range: {common_start} ~ {common_end}")
    print(f"Weather mapping: {out_root / 'weather_station_mapping.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
