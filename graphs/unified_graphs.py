"""
Unified graph protocol + loader — the single source of truth for DMGENet's
four-graph construction across every dataset.

Graphs:
  D: Haversine distance + Gaussian kernel
  N: radius-based binary neighborhood
  S: Jensen-Shannon divergence over target distributions
  F: OpenStreetMap POI cosine similarity (functional/semantic graph)

Keeping hyperparameters centralised here ensures that each dataset's
D/N/S/F graphs are built identically, so experiments remain comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


UNIFIED_GRAPH_PROTOCOL = {
    "D": {"sigma_km": 16.0, "threshold": 0.4},
    "N": {"radius_km": 45.0},
    "S": {"sigma": 0.2, "threshold": 0.6},
    "F": {
        "source": "osm_poi",
        "radius_m": 500,
        "threshold": 0.65,
        "log_scale": True,
        "n_dims": 13,
        "dim_names": [
            "catering",
            "scenic_spots",
            "public_facilities",
            "companies",
            "shopping",
            "transportation",
            "financial",
            "science_education",
            "housing",
            "life_services",
            "sports_leisure",
            "medical",
            "government",
        ],
    },
}


def resolve_graph_dir(root: Path, dataset: str, graph_dir: Optional[str | Path] = None) -> Path:
    if graph_dir is None:
        return root / "dataset" / dataset / "graphs"
    graph_dir = Path(graph_dir)
    return graph_dir if graph_dir.is_absolute() else (root / graph_dir)


def load_graph_tensors(
    root: Path,
    dataset: str,
    device: str | torch.device,
    graph_dir: Optional[str | Path] = None,
) -> Dict[str, torch.Tensor]:
    """Load the four adjacency matrices keyed as Model_D / N / S / POI."""
    graph_path = resolve_graph_dir(root, dataset, graph_dir)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_path}")

    def _load(name: str) -> torch.Tensor:
        path = graph_path / f"{name}_adj.npy"
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        arr = np.load(str(path)).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float).to(device)

    return {
        "Model_D": _load("D"),
        "Model_N": _load("N"),
        "Model_S": _load("S"),
        "Model_POI": _load("F"),
    }


def load_graph_metadata(
    root: Path,
    dataset: str,
    graph_dir: Optional[str | Path] = None,
) -> dict:
    graph_path = resolve_graph_dir(root, dataset, graph_dir)
    meta_path = graph_path / "graphs_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Graph metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))

