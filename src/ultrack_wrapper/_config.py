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

    method: str = "fixed"  # "fixed", "otsu", or "triangle"
    threshold: float = 1.0


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
