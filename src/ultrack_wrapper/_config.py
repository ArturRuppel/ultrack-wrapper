"""Pydantic models for project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Raw data source configuration."""

    ndtiff_path: str
    root_dir: str
    positions: list[int]
    timepoints: Optional[list[int]] = None
    xy_downsample: int = 3


class CellposeConfig(BaseModel):
    """Parameters for Cellpose segmentation (s01a / s01b)."""

    model: str = "nuclei"
    diameter: float = 17.0
    anisotropy: float = 1.0
    min_size: int = 500
    use_gpu: bool = True
    gamma: Optional[float] = None  # None = no correction; 1.0 = identity




class FlowWatershedConfig(BaseModel):
    """Parameters for flow-guided watershed cell segmentation (s02)."""

    flow_scale: float = 1.0
    cellpose_prob_threshold: float = 0.0
    flow_smoothing_sigma: float = 0.0
    method: str = "distance"  # "distance" (fast) or "iterative"


class CellposeContoursConfig(BaseModel):
    """Parameters for cellpose-native contour generation (s02c).

    Uses cellpose.dynamics.compute_masks to generate label maps from flow fields
    and probability maps, then ultrack.utils.labels_to_contours to derive contours.
    """

    cellprob_threshold: float = 0.0
    do_3D: bool = True
    smooth_sigma: float = 0.5
    device: str = "cuda"  # "cuda" for GPU, "cpu" for CPU


class TrackingConfig(BaseModel):
    """Parameters for Ultrack tracking (s03)."""

    # Segmentation hypothesis
    min_area: int = 100
    max_area: int = 1000000
    min_frontier: float = 0.0
    threshold: float = 0.5
    ws_hierarchy: str = "area"  # "area", "dynamics", or "volume"
    anisotropy_penalization: float = 0.0
    n_workers: int = 1

    # Linking
    max_distance: float = 15.0
    max_neighbors: int = 5
    distance_weight: float = 0.0

    # Solver / ILP
    appear_weight: float = -0.001
    disappear_weight: float = -0.001
    division_weight: float = -0.001
    link_function: str = "power"  # "power" or "identity"
    power: float = 4.0
    bias: float = 0.0
    solution_gap: float = 0.001
    time_limit: int = 36000
    window_size: int = 0  # 0 = solve all at once

    # Per-stage overwrite flags (controls whether Run All re-runs each step)
    overwrite_segmentation: bool = True
    overwrite_linking: bool = True
    overwrite_solve: bool = True


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    dataset: DatasetConfig
    cellpose: CellposeConfig = CellposeConfig()
    flow_watershed: FlowWatershedConfig = FlowWatershedConfig()
    cp_contours: CellposeContoursConfig = CellposeContoursConfig()
    tracking: TrackingConfig = TrackingConfig()
