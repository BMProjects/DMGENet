"""
Build the four DMGENet graphs (D, N, S, F) for a dataset and save them under
dataset/{dataset}/graphs/ as .npy files consumed by train_base.py.

Graphs:
  D  — Distance:     Haversine + Gaussian kernel (sigma=16 km, eps=0.4)
  N  — Neighborhood: 0/1 binary, radius threshold (R=45 km)
  S  — Similarity:   Jensen-Shannon over training PM2.5, Gaussian kernel (sigma=0.2, eps=0.6)
  F  — Functional:   OSM POI cosine similarity (13 categories, radius=500m, eps=0.65)

Outputs (written to dataset/{dataset}/graphs/):
  {D,N,S,F}_adj.npy, {D,N,S,F}_edge_index.npy, {D,N,S,F}_edge_weight.npy
  graphs_metadata.json

Example:
  python graphs/build_graphs.py --dataset Beijing_12
  python graphs/build_graphs.py --dataset Chengdu_10

  # Override default hyperparameters:
  python graphs/build_graphs.py --dataset Beijing_12 \\
      --distance-sigma 16 --distance-threshold 0.4 \\
      --neighbor-km 45 \\
      --similarity-sigma 0.2 --similarity-threshold 0.6 \\
      --functional-radius 500 --functional-threshold 0.65
"""

import os
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parent.parent   # DMGENet/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphs.unified_graphs import UNIFIED_GRAPH_PROTOCOL


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two (lat, lon) points."""
    R = 6371.004
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon *distance* (sqrt of JSD), bounded in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Approximate the distributions via histograms over a robust value range.
    all_vals = np.concatenate([p, q])
    bins = min(100, len(all_vals) // 20)
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    if lo >= hi:
        return 0.0
    bin_edges = np.linspace(lo, hi, bins + 1)
    hist_p, _ = np.histogram(p, bins=bin_edges, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()
    m = 0.5 * (hist_p + hist_q)
    jsd = 0.5 * entropy(hist_p, m) + 0.5 * entropy(hist_q, m)
    return math.sqrt(max(0.0, jsd))


def _build_edge_arrays(adj: np.ndarray):
    """Return (edge_index [2, E], edge_weight [E]) from a dense adjacency."""
    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weight = adj[rows, cols].astype(np.float32)
    return edge_index, edge_weight


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_location(location_csv: Path):
    """Return (site_names, lats, lons, station_ids) from a location CSV.

    Column lookup is case-insensitive and accepts the common spellings used
    by both the Chinese (PRSA) and Delhi location files.
    """
    df = pd.read_csv(location_csv)
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    id_col   = col_map.get("station_id") or col_map.get("stationid")
    name_col = col_map.get("station_name") or col_map.get("site_name") or col_map.get("sitename") or df.columns[0]
    lat_col  = col_map.get("latitude")  or col_map.get("lat")
    lon_col  = col_map.get("longitude") or col_map.get("lon")

    if lat_col is None or lon_col is None:
        raise ValueError(f"No lat/lon columns in location.csv: {list(df.columns)}")

    site_names  = df[name_col].tolist()
    lats        = df[lat_col].astype(float).tolist()
    lons        = df[lon_col].astype(float).tolist()
    station_ids = df[id_col].tolist() if id_col else [None] * len(site_names)
    return site_names, lats, lons, station_ids


def _read_pm25_from_csv(fpath: Path) -> np.ndarray:
    """Read a single-station PM2.5 series and mean-impute NaNs."""
    df = pd.read_csv(fpath, index_col=0)
    if "PM2.5" in df.columns:
        vals = df["PM2.5"].values.astype(float)
    elif "PM25" in df.columns:
        vals = df["PM25"].values.astype(float)
    else:
        vals = df.iloc[:, 0].values.astype(float)
    mean_val = np.nanmean(vals) if np.any(np.isfinite(vals)) else 0.0
    return np.where(np.isfinite(vals), vals, mean_val)


def load_pm25_series(aqi_dir: Path, n_stations: int,
                     station_ids: list = None) -> list:
    """Load training-set PM2.5 series. Accepts two filename conventions:

    1. PRSA_Data_{1..N}.csv   (Chinese city standard format)
    2. {station_id}.csv        (Delhi station-code format)
    """
    series_list = []
    for i in range(n_stations):
        fpath = aqi_dir / f"PRSA_Data_{i + 1}.csv"
        if not fpath.exists():
            fpath = aqi_dir / f"PRSA_Data_{i + 1}.csv.gz"
        if not fpath.exists() and station_ids and station_ids[i]:
            fpath = aqi_dir / f"{station_ids[i]}.csv"
        if not fpath.exists() and station_ids and station_ids[i]:
            fpath = aqi_dir / f"{station_ids[i]}.csv.gz"
        if not fpath.exists():
            print(f"  [warn] station {i+1} data not found (tried PRSA_Data_{i+1}.csv"
                  f"{f' and {station_ids[i]}.csv' if station_ids and station_ids[i] else ''}); using zero placeholder")
            series_list.append(np.zeros(100))
            continue
        series_list.append(_read_pm25_from_csv(fpath))
    return series_list


# ──────────────────────────────────────────────────────────────────────────────
# Graph builders
# ──────────────────────────────────────────────────────────────────────────────

def build_distance_graph(lats: list, lons: list,
                         sigma: float = 16.0, threshold: float = 0.4):
    """Distance graph (D): w(i, j) = exp(-d^2 / sigma^2), zeroed below the threshold.

    Self-loops are removed. sigma is expressed in km.
    """
    N = len(lats)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # Gaussian kernel with sigma^2 (not 2*sigma^2) — the unified protocol.
    adj = np.exp(-(dist_mat ** 2) / (sigma ** 2))
    np.fill_diagonal(adj, 0)
    adj[adj <= threshold] = 0.0

    edge_index, edge_weight = _build_edge_arrays(adj)
    n_edges = int((adj > 0).sum())
    print(f"  [D] Distance graph — sigma={sigma}km, eps={threshold}: {n_edges} edges")
    return adj, edge_index, edge_weight


def build_neighbor_graph(lats: list, lons: list, radius_km: float = 45.0):
    """Neighborhood graph (N): edge iff distance ≤ radius_km. Self-loops included."""
    N = len(lats)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        adj[i, i] = 1.0  # self-loop
        for j in range(N):
            if i == j:
                continue
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d <= radius_km:
                adj[i, j] = 1.0

    edge_index, edge_weight = _build_edge_arrays(adj)
    n_edges = int(adj.sum())
    print(f"  [N] Neighborhood graph — R={radius_km}km: {n_edges} edges (incl. {N} self-loops)")
    return adj, edge_index, edge_weight


def build_similarity_graph(pm25_series: list,
                           sigma: float = 0.2, threshold: float = 0.6):
    """Similarity graph (S): Gaussian kernel over pairwise Jensen-Shannon distance.

    Uses JS *distance* (not divergence), computed only on the training split
    to avoid future-information leakage.
    """
    N = len(pm25_series)
    jsd_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = jensen_shannon_distance(pm25_series[i], pm25_series[j])
            jsd_mat[i, j] = d
            jsd_mat[j, i] = d

    adj = np.exp(-(jsd_mat ** 2) / (sigma ** 2))
    np.fill_diagonal(adj, 0)
    adj[adj <= threshold] = 0.0

    edge_index, edge_weight = _build_edge_arrays(adj)
    n_edges = int((adj > 0).sum())
    print(f"  [S] Similarity graph — sigma={sigma}, eps={threshold}: {n_edges} edges")
    return adj, edge_index, edge_weight


def build_functional_graph_osm(lats: list, lons: list,
                                radius: int = 500,
                                threshold: float = 0.65,
                                log_scale: bool = True,
                                delay: float = 1.5,
                                use_cache: bool = True):
    """Functional graph (F) from OpenStreetMap POIs via the Overpass API.

    Pipeline:
      1. Query OSM features within `radius` metres of each station (generic
         building outlines are filtered out in the POI builder).
      2. Project onto 13 urban-function categories (catering, scenic, …).
      3. log(1+x) compress the counts so high-frequency categories don't dominate.
      4. L2 normalise and compute the pairwise cosine similarity matrix.
      5. Apply `threshold` to sparsify.

    Default parameters were validated against the Amap-derived reference matrix
    for the 12-station Beijing grid (Spearman 0.659, Pearson 0.687,
    Edge-F1 0.820 at threshold=0.65; radius=500m + log1p was the best combo).
    """
    try:
        from graphs.build_poi_graph import build_cosine_similarity_matrix
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_poi_graph",
            ROOT / "graphs" / "build_poi_graph.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build_cosine_similarity_matrix = mod.build_cosine_similarity_matrix

    print(f"  [F] OSM functional graph — radius={radius}m, log={log_scale}, eps={threshold}")
    print(f"      Querying Overpass API ({len(lats)} stations)...")

    sim_mat, vectors = build_cosine_similarity_matrix(
        lats, lons,
        radius=radius,
        use_cache=use_cache,
        delay=delay,
        log_scale=log_scale,
    )

    adj = np.where(sim_mat >= threshold, sim_mat, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    edge_index, edge_weight = _build_edge_arrays(adj)
    n_edges = int((adj > 0).sum())
    density = n_edges / (len(lats) * (len(lats) - 1)) * 100
    print(f"  [F] OSM functional graph built: {n_edges} edges (density {density:.1f}%)")

    return adj, edge_index, edge_weight, sim_mat, vectors


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_graph(out_dir: Path, graph_type: str,
               adj: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray):
    """Persist one graph as three .npy files ({type}_adj/edge_index/edge_weight)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{graph_type}_adj.npy",         adj.astype(np.float32))
    np.save(out_dir / f"{graph_type}_edge_index.npy",  edge_index)
    np.save(out_dir / f"{graph_type}_edge_weight.npy", edge_weight.astype(np.float32))


def save_functional_extras(out_dir: Path, sim_mat: np.ndarray, vectors: np.ndarray):
    np.save(out_dir / "F_cosine_sim_raw.npy", sim_mat.astype(np.float32))
    pd.DataFrame(vectors, columns=UNIFIED_GRAPH_PROTOCOL["F"]["dim_names"]).to_csv(
        out_dir / "F_functional_vectors.csv", index=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build the four DMGENet graphs (D/N/S/F) for a dataset"
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name, e.g. Beijing_12, Chengdu_10, Delhi_NCT_Meteo")
    # Path overrides (optional; defaults inferred from dataset/{name}).
    parser.add_argument("--location",   default=None,
                        help="Path to location.csv (default: dataset/{dataset}/location/location.csv)")
    parser.add_argument("--train-pm25", default=None,
                        help="AQI_processed directory (default: dataset/{dataset}/AQI_processed)")
    parser.add_argument("--out-dir",    default=None,
                        help="Output directory (default: dataset/{dataset}/graphs)")
    # Graph hyperparameters.
    parser.add_argument("--distance-sigma",      type=float, default=UNIFIED_GRAPH_PROTOCOL["D"]["sigma_km"],
                        help="Distance-graph Gaussian bandwidth sigma (km) [default: 16.0]")
    parser.add_argument("--distance-threshold",  type=float, default=UNIFIED_GRAPH_PROTOCOL["D"]["threshold"],
                        help="Distance-graph edge threshold [default: 0.4]")
    parser.add_argument("--neighbor-km",         type=float, default=UNIFIED_GRAPH_PROTOCOL["N"]["radius_km"],
                        help="Neighborhood-graph radius R (km) [default: 45.0]")
    parser.add_argument("--similarity-sigma",    type=float, default=UNIFIED_GRAPH_PROTOCOL["S"]["sigma"],
                        help="Similarity-graph sigma [default: 0.2]")
    parser.add_argument("--similarity-threshold",type=float, default=UNIFIED_GRAPH_PROTOCOL["S"]["threshold"],
                        help="Similarity-graph edge threshold [default: 0.6]")
    parser.add_argument("--functional-threshold",type=float, default=UNIFIED_GRAPH_PROTOCOL["F"]["threshold"],
                        help="Functional-graph edge threshold [default: 0.65]")
    parser.add_argument("--functional-radius",   type=int,   default=UNIFIED_GRAPH_PROTOCOL["F"]["radius_m"],
                        help="OSM POI buffer radius in metres [default: 500]")
    parser.add_argument("--no-log-scale",        action="store_true",
                        help="Disable log(1+x) compression of OSM POI counts (enabled by default)")
    args = parser.parse_args()

    dataset_dir = ROOT / "dataset" / args.dataset

    location_csv = Path(args.location) if args.location else dataset_dir / "location" / "location.csv"
    aqi_dir      = Path(args.train_pm25) if args.train_pm25 else dataset_dir / "AQI_processed"
    out_dir      = Path(args.out_dir) if args.out_dir else dataset_dir / "graphs"

    if not location_csv.exists():
        print(f"ERROR: {location_csv} not found")
        print("Run first: python data/build_cn_dataset.py preprocess --city <name>")
        sys.exit(1)
    if not aqi_dir.exists():
        print(f"ERROR: {aqi_dir} not found")
        print("Run first: python data/build_cn_dataset.py preprocess --city <name>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Graph construction: {args.dataset}")
    print(f"{'='*60}")
    print(f"  location   : {location_csv}")
    print(f"  AQI dir    : {aqi_dir}")
    print(f"  output dir : {out_dir}")

    print("\n[1/3] Loading station coordinates...")
    site_names, lats, lons, station_ids = load_location(location_csv)
    N = len(site_names)
    print(f"  {N} stations: {site_names}")

    print("\n[2/3] Loading training PM2.5 series (required by S and F graphs)...")
    pm25_series = load_pm25_series(aqi_dir, N, station_ids=station_ids)
    lengths = [len(s) for s in pm25_series]
    print(f"  PM2.5 series length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")

    print("\n[3/3] Building graphs...")
    adj_D, ei_D, ew_D = build_distance_graph(
        lats, lons,
        sigma=args.distance_sigma, threshold=args.distance_threshold
    )
    adj_N, ei_N, ew_N = build_neighbor_graph(
        lats, lons,
        radius_km=args.neighbor_km
    )
    adj_S, ei_S, ew_S = build_similarity_graph(
        pm25_series,
        sigma=args.similarity_sigma, threshold=args.similarity_threshold
    )
    adj_F, ei_F, ew_F, sim_F, vec_F = build_functional_graph_osm(
        lats, lons,
        radius=args.functional_radius,
        threshold=args.functional_threshold,
        log_scale=(not args.no_log_scale),
    )

    print(f"\nSaving graph files to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_graph(out_dir, "D", adj_D, ei_D, ew_D)
    save_graph(out_dir, "N", adj_N, ei_N, ew_N)
    save_graph(out_dir, "S", adj_S, ei_S, ew_S)
    save_graph(out_dir, "F", adj_F, ei_F, ew_F)
    save_functional_extras(out_dir, sim_F, vec_F)

    metadata = {
        "dataset": args.dataset,
        "num_nodes": N,
        "site_names": site_names,
        "graph_params": {
            "D": {"sigma_km": args.distance_sigma, "threshold": args.distance_threshold,
                  "num_edges": int((adj_D > 0).sum())},
            "N": {"radius_km": args.neighbor_km,
                  "num_edges": int(adj_N.sum())},
            "S": {"sigma": args.similarity_sigma, "threshold": args.similarity_threshold,
                  "num_edges": int((adj_S > 0).sum())},
            "F": {
                "source": UNIFIED_GRAPH_PROTOCOL["F"]["source"],
                "radius_m": args.functional_radius,
                "threshold": args.functional_threshold,
                "log_scale": not args.no_log_scale,
                "n_dims": UNIFIED_GRAPH_PROTOCOL["F"]["n_dims"],
                "dim_names": UNIFIED_GRAPH_PROTOCOL["F"]["dim_names"],
                "num_edges": int((adj_F > 0).sum()),
            },
        }
    }
    meta_path = out_dir / "graphs_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  metadata saved: {meta_path}")

    print(f"\nDone. {N} stations, 4 graph types.")
    print(f"  D edges: {int((adj_D > 0).sum())}")
    print(f"  N edges: {int(adj_N.sum())}")
    print(f"  S edges: {int((adj_S > 0).sum())}")
    print(f"  F edges: {int((adj_F > 0).sum())}")


if __name__ == "__main__":
    main()
