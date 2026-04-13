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


class ForegroundConfig(BaseModel):
    """Parameters for foreground detection (s02)."""

    median_filter: bool = False
    median_radius: int = 2
    gaussian_filter: bool = False
    gaussian_sigma: float = 1.0
    clahe: bool = False
    clahe_clip_limit: float = 0.01
    clahe_kernel_size: int = 0  # 0 = auto (1/8 of image size)
    method: str = "fixed"  # "fixed", "otsu", "triangle", or "sigmoid"
    threshold: float = 1.0
    sigmoid_center: float = 1.0
    sigmoid_steepness: float = 3.0
    fill_holes: bool = True
    fill_holes_max_size: int = 0  # 0 = fill all holes
    morpho_op: str = "none"  # "none", "opening", "closing"
    morpho_radius: int = 2
    remove_small: bool = True
    remove_small_min_size: int = 500
    area_filter: bool = False
    area_filter_min: int = 100
    area_filter_max: int = 100000
    distance_filter: bool = False
    distance_filter_min_radius: float = 3.0


class TrackingConfig(BaseModel):
    """Parameters for Ultrack tracking (s03)."""

    min_area: int = 100
    max_area: int = 100000
    min_frontier: float = 0.0
    max_distance: float = 50.0


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    dataset: DatasetConfig
    cellpose: CellposeConfig = CellposeConfig()
    foreground: ForegroundConfig = ForegroundConfig()
    tracking: TrackingConfig = TrackingConfig()
