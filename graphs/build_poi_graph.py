#!/usr/bin/env python3
"""
Functional (F) graph construction from OpenStreetMap POI data.

For each monitoring station, query OSM features within a radius, map them
to 13 urban-function dimensions, L2-normalise, and compute a pairwise
cosine-similarity matrix thresholded at ε to form DMGENet's F graph.

Outputs (in dataset/<DATASET>/graphs/):
  F_adj.npy                 weighted adjacency (N x N, self-loops = 0)
  F_edge_index.npy          (2, E) int64 edge list
  F_edge_weight.npy         (E,)   float32 weights
  F_cosine_sim_raw.npy      (N x N) similarity matrix before thresholding
  F_functional_vectors.csv  (N x 13) L2-normalised functional vectors

Defaults follow the validated operating point in
graphs/unified_graphs.py::UNIFIED_GRAPH_PROTOCOL["F"] — radius=500 m,
threshold=0.65, log(1+x) scaling, 13-category taxonomy aligned with
the Baidu POI major-category schema (Zhang et al. 2021, PMC8201188).

Example:
  python graphs/build_poi_graph.py build --dataset Beijing_12
  python graphs/build_poi_graph.py build --dataset Chengdu_10 \\
      --radius 500 --threshold 0.65
"""

import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # DMGENet/
CACHE_DIR = ROOT / ".osm_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Public Overpass endpoints — tried in order, retried on timeout.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# ──────────────────────────────────────────────────────────────────────────────
# 13-category urban-function taxonomy.
# Aligned with Zhang et al. 2021 (PMC8201188) Beijing PM2.5 LUR study, which
# uses the Baidu POI major-category schema. Each rule is (OSM_key, value);
# value=None matches any value for that key.
# ──────────────────────────────────────────────────────────────────────────────
FUNCTIONAL_DIMS = {
    "catering": [
        ("amenity", "restaurant"), ("amenity", "cafe"), ("amenity", "fast_food"),
        ("amenity", "bar"), ("amenity", "pub"), ("amenity", "canteen"),
        ("amenity", "food_court"), ("amenity", "ice_cream"), ("amenity", "biergarten"),
    ],
    "scenic_spots": [
        ("tourism", "attraction"), ("tourism", "museum"), ("tourism", "theme_park"),
        ("tourism", "viewpoint"), ("tourism", "zoo"), ("tourism", "aquarium"),
        ("tourism", "gallery"), ("historic", None),
        ("amenity", "theatre"), ("amenity", "cinema"), ("amenity", "arts_centre"),
        ("leisure", "park"), ("leisure", "garden"), ("leisure", "nature_reserve"),
    ],
    "public_facilities": [
        ("amenity", "toilets"), ("amenity", "post_office"),
        ("amenity", "recycling"), ("amenity", "waste_disposal"),
        ("amenity", "water_point"), ("amenity", "telephone"),
        ("man_made", "water_tower"), ("man_made", "wastewater_plant"),
        ("man_made", "street_cabinet"), ("amenity", "shelter"),
    ],
    "companies": [
        ("office", "company"), ("office", "commercial"), ("office", "financial"),
        ("office", "insurance"), ("office", "it"), ("office", "ngo"),
        # building=office is the one building-polygon value kept (see
        # _SKIP_BUILDING_VALUES below): it marks confirmed office buildings.
        ("building", "office"),
        ("office", "architect"), ("office", "consulting"),
    ],
    "shopping": [
        ("shop", None),
        ("amenity", "marketplace"), ("amenity", "supermarket"),
        ("building", "retail"), ("building", "commercial"),
    ],
    "transportation": [
        ("amenity", "fuel"), ("amenity", "parking"), ("amenity", "bus_station"),
        ("amenity", "taxi"), ("amenity", "car_rental"), ("amenity", "bicycle_rental"),
        ("railway", "station"), ("railway", "halt"), ("railway", "subway_entrance"),
        ("aeroway", "terminal"), ("aeroway", "gate"), ("aeroway", "aerodrome"),
        ("public_transport", "station"), ("highway", "bus_stop"),
    ],
    "financial": [
        ("amenity", "bank"), ("amenity", "atm"), ("amenity", "bureau_de_change"),
        ("office", "financial"), ("office", "insurance"), ("office", "accountant"),
    ],
    "science_education": [
        ("amenity", "school"), ("amenity", "university"), ("amenity", "college"),
        ("amenity", "kindergarten"), ("amenity", "library"),
        ("amenity", "research_institute"), ("amenity", "driving_school"),
        ("amenity", "language_school"), ("office", "research"),
        ("building", "university"),
    ],
    "housing": [
        # landuse=residential/commercial is kept (typically 1-3 polygons per
        # buffer); raw building polygons are filtered by _SKIP_BUILDING_VALUES.
        ("landuse", "residential"), ("landuse", "commercial"),
        ("office", "estate_agent"), ("amenity", "community_centre"),
        ("amenity", "social_facility"),
    ],
    "life_services": [
        ("amenity", "laundry"), ("amenity", "dry_cleaning"),
        ("amenity", "vending_machine"), ("amenity", "photo_booth"),
        ("shop", "hairdresser"), ("shop", "beauty"), ("shop", "bakery"),
        ("shop", "convenience"), ("shop", "florist"), ("shop", "optician"),
        ("shop", "dry_cleaning"), ("shop", "tailor"), ("shop", "mobile_phone"),
    ],
    "sports_leisure": [
        ("leisure", "sports_centre"), ("leisure", "stadium"),
        ("leisure", "swimming_pool"), ("leisure", "fitness_centre"),
        ("leisure", "pitch"), ("leisure", "track"), ("leisure", "golf_course"),
        ("leisure", "playground"), ("leisure", "dog_park"),
        ("amenity", "sports_hall"), ("amenity", "swimming_pool"),
        ("sport", None),
    ],
    "medical": [
        ("amenity", "hospital"), ("amenity", "clinic"), ("amenity", "pharmacy"),
        ("amenity", "doctors"), ("amenity", "dentist"), ("amenity", "veterinary"),
        ("healthcare", None),
    ],
    "government": [
        ("office", "government"), ("office", "administrative"), ("office", "military"),
        ("amenity", "townhall"), ("amenity", "police"), ("amenity", "fire_station"),
        ("amenity", "courthouse"), ("amenity", "embassy"), ("amenity", "prison"),
        ("building", "government"),
    ],
}

DIM_NAMES = list(FUNCTIONAL_DIMS.keys())
N_DIMS = len(DIM_NAMES)

# Reverse lookup: (key, value_or_None) → dim index.
_TAG_TO_DIM: dict = {}
for dim_idx, (dim_name, rules) in enumerate(FUNCTIONAL_DIMS.items()):
    for key, val in rules:
        _TAG_TO_DIM[(key, val)] = dim_idx
        if val is None:
            _TAG_TO_DIM[(key, "__any__")] = dim_idx

# Generic building polygons to exclude from counts.
# Chinese cities in OSM often render a single apartment complex as dozens or
# hundreds of separate `building=residential` polygons. Counting them makes
# every suburban station's functional vector collapse to the same housing-
# dominated profile. `building=office` is kept (it survives as a rule above)
# because it typically marks one confirmed office building.
_SKIP_BUILDING_VALUES = frozenset({
    "residential", "apartments", "house", "detached",
    "dormitory", "yes", "static_caravan", "terrace", "semidetached_house",
    "bungalow", "barn", "garage", "garages", "shed", "hut",
})
# Large-area landuse polygons to exclude — one polygon covers a whole region
# and is not a comparable "point of interest". landuse=residential/commercial
# are kept (they appear in the housing dim above, 1-3 per buffer).
_SKIP_LANDUSE_VALUES = frozenset({
    "grass", "meadow", "forest", "farmland", "farm",
    "orchard", "vineyard", "cemetery", "military",
    "railway", "basin", "reservoir", "quarry",
    "allotments", "brownfield", "greenfield", "landfill",
})


# ──────────────────────────────────────────────────────────────────────────────
# OSM query + classification
# ──────────────────────────────────────────────────────────────────────────────

def _overpass_query(lat: float, lon: float, radius: int,
                    use_cache: bool = True) -> list:
    """Query Overpass for tagged OSM elements within `radius` metres.

    Results are cached under .osm_cache/ keyed by (lat, lon, radius) so
    repeated builds and rate-limit-induced retries do not re-hit the API.
    """
    cache_key = hashlib.md5(f"{lat:.5f}_{lon:.5f}_{radius}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"osm_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Nodes + ways on the main tag keys (relations excluded to keep queries
    # light). `building` and `railway` are filtered to a narrow value set to
    # avoid pulling every indoor room or maintenance track.
    query_keys = [
        "amenity", "shop", "tourism", "leisure", "historic",
        "office", "healthcare", "landuse", "man_made", "aeroway",
    ]
    building_vals = (
        "residential|apartments|dormitory|house|detached|"
        "retail|commercial|office|warehouse|factory|hotel|"
        "church|mosque|temple|cathedral"
    )
    railway_vals = "station|halt|subway_entrance"

    node_way_blocks = "\n".join(
        f'  node(around:{radius},{lat},{lon})["{k}"];\n'
        f'  way(around:{radius},{lat},{lon})["{k}"];'
        for k in query_keys
    )
    node_way_blocks += (
        f'\n  node(around:{radius},{lat},{lon})'
        f'[building~"^({building_vals})$"];\n'
        f'  way(around:{radius},{lat},{lon})'
        f'[building~"^({building_vals})$"];\n'
        f'  node(around:{radius},{lat},{lon})'
        f'[railway~"^({railway_vals})$"];\n'
        f'  way(around:{radius},{lat},{lon})'
        f'[railway~"^({railway_vals})$"];'
    )

    overpass_ql = (
        f"[out:json][timeout:90];\n"
        f"(\n{node_way_blocks}\n);\n"
        f"out tags center;"
    )

    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(3):
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": overpass_ql},
                    timeout=120,
                    headers={"User-Agent": "DMGENet-research/1.0"},
                )
                resp.raise_for_status()
                text = resp.text.strip()
                if not text:
                    raise ValueError("empty response body")
                data = json.loads(text)
                elements = data.get("elements", [])
                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(elements, f)
                return elements
            except Exception as e:
                last_err = e
                # Exponential backoff: 3, 6, 12 seconds.
                time.sleep(3 * (2 ** attempt))
        time.sleep(5)

    print(f"  [warn] Overpass query failed ({lat:.4f},{lon:.4f} r={radius}m): {last_err}")
    return []


def _classify_element(tags: dict) -> list:
    """Map one OSM element's tags to the set of functional-dim indices it matches.

    An element may hit multiple dimensions (e.g. a hotel with a restaurant).
    Generic building polygons and large-area landuse values are filtered out
    first — see the rationale on `_SKIP_BUILDING_VALUES` / `_SKIP_LANDUSE_VALUES`.
    """
    if tags.get("building", "") in _SKIP_BUILDING_VALUES:
        return []
    if tags.get("landuse", "") in _SKIP_LANDUSE_VALUES:
        return []

    matched = set()
    for key, val in tags.items():
        if (key, val) in _TAG_TO_DIM:
            matched.add(_TAG_TO_DIM[(key, val)])
        if (key, "__any__") in _TAG_TO_DIM:
            matched.add(_TAG_TO_DIM[(key, "__any__")])
    return list(matched)


def build_functional_vector(lat: float, lon: float, radius: int,
                            use_cache: bool = True,
                            log_scale: bool = True,
                            min_elements: int = 5,
                            fallback_radii: tuple = (750, 1000, 1500)) -> np.ndarray:
    """Build a single station's 13-dim L2-normalised functional vector.

    Args:
      radius:         primary buffer radius (metres).
      log_scale:      apply log(1+x) compression to raw counts. Analogous to
                      TF-IDF term-frequency damping; suppresses high-frequency
                      OSM categories (e.g. shops) that would otherwise drown
                      out rarer but more informative ones.
      min_elements:   minimum effective POI count. Below this, the function
                      auto-expands through `fallback_radii` until it finds a
                      radius with enough POIs — protects sparse suburban
                      stations from producing a degenerate zero vector.
    """
    def _count_elements(elements):
        counts = np.zeros(N_DIMS, dtype=float)
        for elem in elements:
            for dim_idx in _classify_element(elem.get("tags", {})):
                counts[dim_idx] += 1.0
        return counts

    elements = _overpass_query(lat, lon, radius, use_cache=use_cache)
    counts = _count_elements(elements)
    effective = int(counts.sum())

    used_radius = radius
    if effective < min_elements:
        for fr in fallback_radii:
            if fr <= radius:
                continue
            fb_counts = _count_elements(
                _overpass_query(lat, lon, fr, use_cache=use_cache)
            )
            if fb_counts.sum() >= min_elements:
                counts = fb_counts
                used_radius = fr
                break

    if used_radius != radius:
        print(f"    [auto-expand] ({lat:.4f},{lon:.4f}) r={radius}m→{used_radius}m "
              f"(effective POI: {effective}→{int(counts.sum())})")

    if log_scale:
        counts = np.log1p(counts)

    norm = np.linalg.norm(counts)
    if norm > 1e-8:
        counts = counts / norm
    return counts


def build_cosine_similarity_matrix(lats: list, lons: list,
                                   radius: int, use_cache: bool = True,
                                   delay: float = 1.0,
                                   log_scale: bool = True):
    """Build the N×N cosine-similarity matrix (diag=1, values in [0, 1])."""
    N = len(lats)
    vectors = np.zeros((N, N_DIMS), dtype=float)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        vectors[i] = build_functional_vector(
            lat, lon, radius, use_cache=use_cache, log_scale=log_scale
        )
        if i < N - 1:
            time.sleep(delay)

    sim = vectors @ vectors.T

    # Correct floating-point drift; non-negative vectors cannot produce < 0.
    np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, 0.0, 1.0)

    return sim, vectors


# ──────────────────────────────────────────────────────────────────────────────
# Build path: produce an F graph for a given dataset
# ──────────────────────────────────────────────────────────────────────────────

def run_build(args):
    """Generate and save the OSM-based F graph for `args.dataset`."""
    print(f"\n{'='*60}")
    print(f"OSM-based F graph: {args.dataset}")
    print(f"  radius={args.radius}m  threshold={args.threshold}")
    print(f"{'='*60}")

    dataset_dir = ROOT / "dataset" / args.dataset
    loc_csv = dataset_dir / "location" / "location.csv"
    out_dir = dataset_dir / "graphs"

    if not loc_csv.exists():
        print(f"ERROR: location file not found: {loc_csv}")
        sys.exit(1)

    df_loc = pd.read_csv(loc_csv)
    col_map = {c.lower(): c for c in df_loc.columns}
    name_col = col_map.get("site_name") or col_map.get("sitename") or df_loc.columns[0]
    lat_col  = col_map.get("latitude")  or col_map.get("lat")
    lon_col  = col_map.get("longitude") or col_map.get("lon")

    station_names = df_loc[name_col].tolist()
    lats = df_loc[lat_col].astype(float).tolist()
    lons = df_loc[lon_col].astype(float).tolist()
    N = len(station_names)
    print(f"{N} stations: {station_names}")

    log_scale = not getattr(args, "no_log_scale", False)
    sim_mat, vectors = build_cosine_similarity_matrix(
        lats, lons,
        radius=args.radius,
        use_cache=(not args.no_cache),
        delay=args.delay,
        log_scale=log_scale,
    )

    # Apply threshold; zero the diagonal so the F graph has no self-loops
    # (consistent with D / N / S graphs).
    adj = np.where(sim_mat >= args.threshold, sim_mat, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    rows, cols = np.where(adj > 0)
    edge_index  = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weight = adj[rows, cols]
    n_edges = int((adj > 0).sum())

    print(f"\nFunctional vectors ({N} x {N_DIMS}):")
    print(f"{'station':<20}" + "".join(f"{d[:5]:>8}" for d in DIM_NAMES))
    for i, sn in enumerate(station_names):
        print(f"{sn:<20}" + "".join(f"{vectors[i,j]:8.3f}" for j in range(N_DIMS)))

    print(f"\nThresholded adjacency (ε={args.threshold}):")
    print(f"  edges={n_edges}  density={n_edges/(N*(N-1))*100:.1f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "F_adj.npy",         adj)
    np.save(out_dir / "F_edge_index.npy",  edge_index)
    np.save(out_dir / "F_edge_weight.npy", edge_weight)
    np.save(out_dir / "F_cosine_sim_raw.npy", sim_mat.astype(np.float32))

    df_vec = pd.DataFrame(vectors, columns=DIM_NAMES, index=station_names)
    df_vec.to_csv(out_dir / "F_functional_vectors.csv")

    meta_path = out_dir / "graphs_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.setdefault("graph_params", {})["F"] = {
            "source": "osm_poi",
            "radius_m": args.radius,
            "threshold": args.threshold,
            "log_scale": log_scale,
            "n_dims": N_DIMS,
            "dim_names": DIM_NAMES,
            "num_edges": n_edges,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nF graph saved to {out_dir}")
    print(f"  F_adj.npy:                {adj.shape}")
    print(f"  F_edge_index.npy:         {edge_index.shape}")
    print(f"  F_edge_weight.npy:        {edge_weight.shape}")
    print(f"  F_functional_vectors.csv  ({N_DIMS}-dim functional vectors)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build the DMGENet F graph from OpenStreetMap POIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_bld = sub.add_parser("build", help="Build the F graph for a dataset.")
    p_bld.add_argument("--dataset", required=True,
                       help="Dataset name (e.g. Beijing_12, Chengdu_10).")
    p_bld.add_argument("--radius", type=int, default=500,
                       help="Buffer radius in metres (default: 500).")
    p_bld.add_argument("--threshold", type=float, default=0.65,
                       help="Adjacency threshold ε (default: 0.65).")
    p_bld.add_argument("--no-cache", action="store_true",
                       help="Ignore the local Overpass cache.")
    p_bld.add_argument("--no-log-scale", action="store_true",
                       help="Disable log(1+x) count compression.")
    p_bld.add_argument("--delay", type=float, default=1.5,
                       help="Seconds between Overpass requests (default: 1.5).")

    args = parser.parse_args()
    if args.mode == "build":
        run_build(args)


if __name__ == "__main__":
    main()
