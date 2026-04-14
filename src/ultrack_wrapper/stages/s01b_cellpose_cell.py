"""s01b — Cellpose 2D cell segmentation.

Runs Cellpose 2D segmentation on Z-projected cell stack (two-channel zavg).

Inputs (per position)
---------------------
  0_raw/cell_zavg_405.tif       (T, H, W)  uint16  — nuclear marker (405 nm)
  0_raw/cell_zavg_488.tif       (T, H, W)  uint16  — membrane marker (488 nm)

Outputs (per position)
----------------------
  1b_cellpose_cell/
    run_params.json
    cell_dp.tif                  (T, 2, H, W)  float32  — flow fields
    cell_prob.tif                (T, H, W)     float32  — probability map
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from ultrack_wrapper._config import CellposeConfig
from ultrack_wrapper._paths import cell_zavg_path, cellpose_cell_dir


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_model(model_type: str, use_gpu: bool):
    from cellpose.models import CellposeModel

    gpu = use_gpu
    if gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA not available — falling back to CPU", flush=True)
                gpu = False
            else:
                print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        except ImportError:
            print("torch not importable — falling back to CPU", flush=True)
            gpu = False

    model = CellposeModel(gpu=gpu, pretrained_model=model_type)
    print(f"Model '{model_type}' loaded  (gpu={gpu})", flush=True)
    return model


# ── Core run function ─────────────────────────────────────────────────────────


def run(
    root_dir: str | Path,
    pos: int,
    cfg: CellposeConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Run Cellpose 2D segmentation on cell zavg stack for one position.

    Yields ``(done, total, label)`` progress tuples.
    """
    root_dir = Path(root_dir)

    # Load input channels
    path_405 = cell_zavg_path(root_dir, pos, 405)
    path_488 = cell_zavg_path(root_dir, pos, 488)

    if not path_405.exists():
        print(f"[error] Input not found: {path_405}", file=sys.stderr)
        return
    if not path_488.exists():
        print(f"[error] Input not found: {path_488}", file=sys.stderr)
        return

    stack_405 = tifffile.imread(str(path_405))  # (T, H, W) uint16
    stack_488 = tifffile.imread(str(path_488))  # (T, H, W) uint16

    if stack_405.ndim != 3 or stack_488.ndim != 3:
        print(
            f"[error] Expected (T, H, W), got {stack_405.shape} and {stack_488.shape}",
            file=sys.stderr,
        )
        return
    if stack_405.shape != stack_488.shape:
        print(
            f"[error] Channel shapes must match: {stack_405.shape} vs {stack_488.shape}",
            file=sys.stderr,
        )
        return

    T, H, W = stack_405.shape
    print(f"pos{pos:02d}  input shape={stack_405.shape}  dtype={stack_405.dtype}", flush=True)
    print(f"  T={T}  H={H}  W={W}", flush=True)
    if cfg.gamma is not None:
        print(f"  gamma={cfg.gamma}", flush=True)

    # Check outputs
    out_dir = cellpose_cell_dir(root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_path = out_dir / "cell_dp.tif"
    prob_path = out_dir / "cell_prob.tif"

    if not overwrite and dp_path.exists() and prob_path.exists():
        print(f"pos{pos:02d}: all outputs exist — skipping.", flush=True)
        return

    run_params_path = out_dir / "run_params.json"
    if not run_params_path.exists():
        run_params_path.write_text(
            json.dumps(
                {
                    "stage": "cellpose_cell",
                    "pos": pos,
                    "params": cfg.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Load model
    model = _load_model(cfg.model, cfg.use_gpu)

    print(
        f"  diameter={cfg.diameter}  min_size={cfg.min_size}",
        flush=True,
    )

    dp_list: list[np.ndarray] = []
    prob_list: list[np.ndarray] = []

    for t in range(T):
        # Reorder to (H, W, 2): cytoplasm (488nm) first, nucleus (405nm) second
        img = np.stack([stack_488[t], stack_405[t]], axis=-1).astype(np.float32)  # (H, W, 2)

        # Optional gamma correction
        gamma = cfg.gamma
        if gamma is not None and gamma != 1.0:
            for c in range(img.shape[2]):
                ch = img[:, :, c]
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min)
                    img[:, :, c] = (ch_norm ** gamma) * (ch_max - ch_min) + ch_min

        print(f"  [{t+1:3d}/{T}]  t{t:03d} ...", end="", flush=True)

        _, flows, _ = model.eval(
            img,
            diameter=cfg.diameter if cfg.diameter > 0 else None,
            min_size=cfg.min_size,
        )
        # flows[1]: dP  (2, H, W) float32
        # flows[2]: cellprob  (H, W) float32
        dp_list.append(flows[1].astype(np.float32))
        prob_list.append(flows[2].astype(np.float32))
        print("  done", flush=True)

    print("Assembling stacks ...", flush=True)
    stack_dp = np.stack(dp_list, axis=0)  # (T, 2, H, W)
    stack_prob = np.stack(prob_list, axis=0)  # (T, H, W)

    tifffile.imwrite(
        str(dp_path),
        stack_dp,
        compression="zlib",
        metadata={"axes": "TCYX"},
    )
    tifffile.imwrite(
        str(prob_path),
        stack_prob,
        compression="zlib",
        metadata={"axes": "TYX"},
    )

    print(f"  → {dp_path.name}  {stack_dp.shape}  {stack_dp.dtype}", flush=True)
    print(f"  → {prob_path.name}  {stack_prob.shape}  {stack_prob.dtype}", flush=True)
    print("Done.", flush=True)


# ── CLI entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="s01b — Cellpose 2D cell segmentation",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root project directory",
    )
    parser.add_argument(
        "--pos",
        required=True,
        type=int,
        help="Position index",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with CellposeConfig fields (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = CellposeConfig(**cfg_dict)

    for done, total, label in run(args.root_dir, args.pos, cfg, overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)
