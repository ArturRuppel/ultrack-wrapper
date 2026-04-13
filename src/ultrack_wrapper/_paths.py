"""Path resolution for project directories and stage inputs/outputs."""

from __future__ import annotations

from pathlib import Path


# ── Stage folder prefixes ────────────────────────────────────────────────────

_STAGE_PREFIX: dict[str, str] = {
    "raw": "0_raw",
    "cellpose_nucleus": "1a_cellpose_nucleus",
    "cellpose_cell": "1b_cellpose_cell",
    "foreground": "2_foreground",
    "contours": "2b_contours",
    "tracking": "3_tracking",
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


def cell_zavg_path(root_dir: str | Path, pos: int) -> Path:
    return raw_dir(root_dir, pos) / "cell_zavg.tif"


# ── Cellpose (s01) ──────────────────────────────────────────────────────────


def cellpose_nucleus_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "cellpose_nucleus", pos)


def cellpose_cell_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "cellpose_cell", pos)


# ── Foreground (s02) ────────────────────────────────────────────────────────


def foreground_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "foreground", pos)


# ── Contours (s02b) ────────────────────────────────────────────────────────


def contours_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "contours", pos)


# ── Tracking (s03) ──────────────────────────────────────────────────────────


def tracking_dir(root_dir: str | Path, pos: int) -> Path:
    return stage_dir(root_dir, "tracking", pos)
