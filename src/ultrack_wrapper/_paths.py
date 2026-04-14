"""Path resolution for project directories and stage inputs/outputs."""

from __future__ import annotations

from pathlib import Path


# ── Stage folder prefixes ────────────────────────────────────────────────────

_STAGE_PREFIX: dict[str, str] = {
    "raw": "0_raw",
    "cellpose_nucleus": "1a_cellpose_nucleus",
    "cellpose_cell": "1b_cellpose_cell",
    "tracking": "2_tracking",
    "cell_segmentation": "3_cell_segmentation",
    "foreground": "2_foreground",
    "contours": "2b_contours",
    "flow_watershed": "2c_flow_watershed",
    "proj2d": "4_proj2d",
    "cell_labels": "5_cell_labels",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def pos_dir(root_dir: str | Path, pos: int) -> Path:
    return Path(root_dir) / f"pos{pos:02d}"


def stage_dir(root_dir: str | Path, stage_name: str, pos: int) -> Path:
    return pos_dir(root_dir, pos) / _STAGE_PREFIX[stage_name]


# ── Raw (s00) ────────────────────────────────────────────────────────────────


def raw_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "raw", pos)


def nucleus_3d_path(root_dir: str | Path, pos: int, t: int) -> Path:
    return raw_dir(root_dir, pos) / f"nucleus_3d_t{t:03d}.tif"


def cell_zavg_path(root_dir: str | Path, pos: int, channel: int = None) -> Path:
    """Get path for zavg file.

    If channel is specified (405 or 488), returns channel-specific file.
    Otherwise returns the legacy combined file path (for backwards compatibility).
    """
    if channel == 405:
        return raw_dir(root_dir, pos) / "cell_zavg_405.tif"
    elif channel == 488:
        return raw_dir(root_dir, pos) / "cell_zavg_488.tif"
    return raw_dir(root_dir, pos) / "cell_zavg.tif"


# ── Cellpose (s01) ──────────────────────────────────────────────────────────


def cellpose_nucleus_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "cellpose_nucleus", pos)


def cellpose_cell_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "cellpose_cell", pos)


# ── Cell Segmentation (s03) ────────────────────────────────────────────────


def cell_segmentation_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "cell_segmentation", pos)


# ── Foreground (s02) ────────────────────────────────────────────────────────


def foreground_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "foreground", pos)


# ── Contours (s02b) ────────────────────────────────────────────────────────


def contours_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "contours", pos)


# ── Flow Watershed (s02c) ──────────────────────────────────────────────────────


def flow_watershed_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "flow_watershed", pos)


# ── Tracking (s03) ──────────────────────────────────────────────────────────


def tracking_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "tracking", pos)
