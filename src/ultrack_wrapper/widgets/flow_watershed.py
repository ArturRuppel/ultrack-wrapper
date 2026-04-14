"""Flow-guided watershed cell segmentation widget."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari.qt.threading import thread_worker

from ultrack_wrapper._paths import (
    cellpose_cell_dir,
    cellpose_nucleus_dir,
    cell_segmentation_dir,
)


class FlowWatershedConfig:
    """Configuration for flow watershed segmentation."""

    def __init__(
        self,
        flow_scale: float = 1.0,
        cellpose_prob_threshold: float = 0.0,
        flow_smoothing_sigma: float = 0.0,
        max_iterations: int = 50,
        uniform_growth_rate: float = 0.2,
        opening_radius: int = 1,
        closing_radius: int = 1,
        boundary_smoothness: float = 0.5,
        fill_holes_threshold: float = 0.5,
    ):
        self.flow_scale = flow_scale
        self.cellpose_prob_threshold = cellpose_prob_threshold
        self.flow_smoothing_sigma = flow_smoothing_sigma
        self.max_iterations = max_iterations
        self.uniform_growth_rate = uniform_growth_rate
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.boundary_smoothness = boundary_smoothness
        self.fill_holes_threshold = fill_holes_threshold

    def model_dump(self) -> dict:
        return {
            "flow_scale": self.flow_scale,
            "cellpose_prob_threshold": self.cellpose_prob_threshold,
            "flow_smoothing_sigma": self.flow_smoothing_sigma,
            "max_iterations": self.max_iterations,
            "uniform_growth_rate": self.uniform_growth_rate,
            "opening_radius": self.opening_radius,
            "closing_radius": self.closing_radius,
            "boundary_smoothness": self.boundary_smoothness,
            "fill_holes_threshold": self.fill_holes_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FlowWatershedConfig:
        # Drop 'method' if present in loaded config for backwards compatibility
        data.pop("method", None)
        return cls(**data)


def _load_nuclear_labels(root_dir: Path | str, pos: int) -> np.ndarray | None:
    """Load tracked nuclear labels from 2_ultrack."""
    try:
        # Try to load from ultrack output
        tracking_labels_path = (
            Path(root_dir) / f"pos{pos:02d}" / "2_ultrack" / "tracked_labels_proj2d_corrected.tif"
        )
        if tracking_labels_path.exists():
            return tifffile.imread(str(tracking_labels_path)).astype(np.int32)
    except Exception:
        pass
    return None


def _load_cellpose_data(root_dir: Path | str, pos: int, t: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load cellpose flow and probability from s1b for a given timepoint."""
    try:
        cell_dir = cellpose_cell_dir(root_dir, pos)

        # Load flow field (if available)
        flow_path = cell_dir / "cell_dp.tif"
        flow = None
        if flow_path.exists():
            flow = tifffile.imread(str(flow_path)).astype(np.float32)
            # Handle shape: cellpose outputs (T, 2, H, W), need to transpose to (T, H, W, 2) or (H, W, 2)
            if flow.ndim == 4:
                # Transpose from (T, 2, H, W) to (T, H, W, 2)
                flow = np.transpose(flow, (0, 2, 3, 1))
                flow = flow[t]  # Get timepoint t (now shape H, W, 2)
            elif flow.ndim == 3:
                # If already single timepoint, transpose from (2, H, W) to (H, W, 2)
                flow = np.transpose(flow, (1, 2, 0))
            else:
                flow = None

        # Load probability field
        prob_path = cell_dir / "cell_prob.tif"
        prob = None
        if prob_path.exists():
            prob = tifffile.imread(str(prob_path)).astype(np.float32)
            if prob.ndim == 3:
                prob = prob[t]  # Get timepoint t

        return flow, prob
    except Exception:
        return None, None


def run_flow_watershed(
    root_dir: str | Path,
    pos: int,
    config: FlowWatershedConfig,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Run flow watershed segmentation for a full stack.

    Args:
        progress_callback: Optional callable(current, total) to report progress

    Returns (nuclear_labels_stack, cell_labels_stack) or None on error.
    """
    from ultrack_wrapper.processing.flow_watershed import flow_guided_watershed

    root_dir = Path(root_dir)

    # Load nuclear labels
    nuclear_labels = _load_nuclear_labels(root_dir, pos)
    if nuclear_labels is None:
        print("Could not load nuclear labels")
        return None

    if nuclear_labels.ndim != 3:
        print(f"Expected (T, H, W), got {nuclear_labels.shape}")
        return None

    T = nuclear_labels.shape[0]
    cell_labels_stack = []

    # Load cellpose data (same for all timepoints or per-timepoint)
    flow_full, prob_full = _load_cellpose_data(root_dir, pos, 0)
    if flow_full is None:
        print(f"Could not load cellpose flow")
        return None

    # Process each timepoint
    for t in range(T):
        try:
            nuc_t = nuclear_labels[t]

            # Get flow/prob for this timepoint
            if flow_full.ndim == 4 and flow_full.shape[0] == T:
                flow_t = flow_full[t]
            else:
                flow_t = flow_full

            if prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == T:
                prob_t = prob_full[t]
            else:
                prob_t = prob_full

            # Run segmentation
            from ultrack_wrapper.processing.flow_watershed_postprocessing import postprocess_flow_watershed

            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=config.flow_scale,
                cellpose_prob_threshold=config.cellpose_prob_threshold,
                flow_smoothing_sigma=config.flow_smoothing_sigma,
                max_iterations=config.max_iterations,
                uniform_growth_rate=config.uniform_growth_rate,
            )

            # Apply post-processing
            cell_labels = postprocess_flow_watershed(
                cell_labels,
                cellpose_prob=prob_t,
                opening_radius=config.opening_radius,
                closing_radius=config.closing_radius,
                boundary_smoothness=config.boundary_smoothness,
                fill_holes_threshold=config.fill_holes_threshold,
            )

            cell_labels_stack.append(cell_labels)

        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        # Report progress
        if progress_callback is not None:
            progress_callback(t + 1, T)

    stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
    return nuclear_labels, stack


class FlowWatershedWidget(QWidget):
    """Widget for flow-guided watershed cell segmentation."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Root directory ───────────────────────────────────────────────
        layout.addWidget(QLabel("Root project directory"))
        row = QHBoxLayout()
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("/path/to/project")
        row.addWidget(self._root_edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_root)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Position ─────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Position"))
        self._pos_spin = QSpinBox()
        self._pos_spin.setRange(0, 1000)
        self._pos_spin.setValue(0)
        row.addWidget(self._pos_spin)
        row.addStretch()
        layout.addLayout(row)

        # ── Frame selector and preview ───────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview frame"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 1000)
        self._frame_spin.setValue(0)
        row.addWidget(self._frame_spin)
        preview_btn = QPushButton("Preview Frame")
        preview_btn.clicked.connect(self._on_preview_frame)
        row.addWidget(preview_btn)
        row.addStretch()
        layout.addLayout(row)

        # ── Parameters ───────────────────────────────────────────────────
        param_group = QGroupBox("Parameters")
        pg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Flow scale (blend factor)"))
        self._flow_scale_spin = QDoubleSpinBox()
        self._flow_scale_spin.setRange(0.0, 3.0)
        self._flow_scale_spin.setSingleStep(0.1)
        self._flow_scale_spin.setDecimals(2)
        self._flow_scale_spin.setValue(1.0)
        row.addWidget(self._flow_scale_spin)
        pg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cellpose prob threshold"))
        self._prob_threshold_spin = QDoubleSpinBox()
        self._prob_threshold_spin.setRange(-100.0, 100.0)
        self._prob_threshold_spin.setSingleStep(1.0)
        self._prob_threshold_spin.setDecimals(1)
        self._prob_threshold_spin.setValue(0.0)
        row.addWidget(self._prob_threshold_spin)
        pg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Flow smoothing (σ)"))
        self._smoothing_spin = QDoubleSpinBox()
        self._smoothing_spin.setRange(0.0, 5.0)
        self._smoothing_spin.setSingleStep(0.1)
        self._smoothing_spin.setDecimals(2)
        self._smoothing_spin.setValue(0.0)
        row.addWidget(self._smoothing_spin)
        pg_layout.addLayout(row)


        row = QHBoxLayout()
        row.addWidget(QLabel("Max iterations"))
        self._max_iterations_spin = QSpinBox()
        self._max_iterations_spin.setRange(1, 2000)
        self._max_iterations_spin.setSingleStep(10)
        self._max_iterations_spin.setValue(50)
        row.addWidget(self._max_iterations_spin)
        row.addStretch()
        pg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Uniform growth rate"))
        self._uniform_growth_spin = QDoubleSpinBox()
        self._uniform_growth_spin.setRange(0.0, 1.0)
        self._uniform_growth_spin.setSingleStep(0.05)
        self._uniform_growth_spin.setDecimals(2)
        self._uniform_growth_spin.setValue(0.2)
        row.addWidget(self._uniform_growth_spin)
        row.addStretch()
        pg_layout.addLayout(row)

        param_group.setLayout(pg_layout)
        layout.addWidget(param_group)

        # ── Post-processing Parameters ──────────────────────────────────
        postproc_group = QGroupBox("Post-processing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Opening radius"))
        self._opening_radius_spin = QSpinBox()
        self._opening_radius_spin.setRange(0, 10)
        self._opening_radius_spin.setValue(1)
        row.addWidget(self._opening_radius_spin)
        row.addStretch()
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Closing radius"))
        self._closing_radius_spin = QSpinBox()
        self._closing_radius_spin.setRange(0, 10)
        self._closing_radius_spin.setValue(1)
        row.addWidget(self._closing_radius_spin)
        row.addStretch()
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Boundary smoothness"))
        self._boundary_smooth_spin = QDoubleSpinBox()
        self._boundary_smooth_spin.setRange(0.0, 1.0)
        self._boundary_smooth_spin.setSingleStep(0.05)
        self._boundary_smooth_spin.setDecimals(2)
        self._boundary_smooth_spin.setValue(0.5)
        row.addWidget(self._boundary_smooth_spin)
        row.addStretch()
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Prob trim threshold"))
        self._prob_trim_spin = QDoubleSpinBox()
        self._prob_trim_spin.setRange(0.0, 1.0)
        self._prob_trim_spin.setSingleStep(0.05)
        self._prob_trim_spin.setDecimals(2)
        self._prob_trim_spin.setValue(0.5)
        row.addWidget(self._prob_trim_spin)
        row.addStretch()
        pp_layout.addLayout(row)

        postproc_group.setLayout(pp_layout)
        layout.addWidget(postproc_group)

        # ── Process button ───────────────────────────────────────────────
        self._run_btn = QPushButton("Process Stack")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # ── Overwrite ────────────────────────────────────────────────────
        self._overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self._overwrite_check)

        # ── Load results ─────────────────────────────────────────────────
        self._load_btn = QPushButton("Load Results")
        self._load_btn.clicked.connect(self._on_load_results)
        layout.addWidget(self._load_btn)

        # ── Save / Load parameters ───────────────────────────────────────
        row = QHBoxLayout()
        save_btn = QPushButton("Save Parameters…")
        save_btn.clicked.connect(self._on_save_params)
        row.addWidget(save_btn)
        load_btn = QPushButton("Load Parameters…")
        load_btn.clicked.connect(self._on_load_params)
        row.addWidget(load_btn)
        layout.addLayout(row)

        # ── Progress ─────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self.setLayout(layout)

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select root project directory (parent of pos00, pos01, etc.)"
        )
        if d:
            d_path = Path(d)
            if d_path.name.startswith("pos") and d_path.name[3:].isdigit():
                d = str(d_path.parent)
            self._root_edit.setText(d)

    def _build_config(self) -> FlowWatershedConfig:
        return FlowWatershedConfig(
            flow_scale=self._flow_scale_spin.value(),
            cellpose_prob_threshold=self._prob_threshold_spin.value(),
            flow_smoothing_sigma=self._smoothing_spin.value(),
            max_iterations=self._max_iterations_spin.value(),
            uniform_growth_rate=self._uniform_growth_spin.value(),
            opening_radius=self._opening_radius_spin.value(),
            closing_radius=self._closing_radius_spin.value(),
            boundary_smoothness=self._boundary_smooth_spin.value(),
            fill_holes_threshold=self._prob_trim_spin.value(),
        )

    def _apply_config(self, cfg: FlowWatershedConfig) -> None:
        self._flow_scale_spin.setValue(cfg.flow_scale)
        self._prob_threshold_spin.setValue(cfg.cellpose_prob_threshold)
        self._smoothing_spin.setValue(cfg.flow_smoothing_sigma)
        self._max_iterations_spin.setValue(cfg.max_iterations)
        self._uniform_growth_spin.setValue(cfg.uniform_growth_rate)
        self._opening_radius_spin.setValue(cfg.opening_radius)
        self._closing_radius_spin.setValue(cfg.closing_radius)
        self._boundary_smooth_spin.setValue(cfg.boundary_smoothness)
        self._prob_trim_spin.setValue(cfg.fill_holes_threshold)

    def _on_preview_frame(self) -> None:
        root_dir = self._root_edit.text().strip()
        if not root_dir:
            self._status_label.setText("Set root directory first.")
            return

        pos = int(self._pos_spin.value())
        frame = int(self._frame_spin.value())
        cfg = self._build_config()

        self._status_label.setText(f"Processing frame {frame}…")

        try:
            from ultrack_wrapper.processing.flow_watershed import flow_guided_watershed
            root_dir_path = Path(root_dir)

            # Load nuclear labels
            nuclear_labels = _load_nuclear_labels(root_dir_path, pos)
            if nuclear_labels is None:
                self._status_label.setText("Could not load nuclear labels.")
                return

            # Load cellpose data
            flow_full, prob_full = _load_cellpose_data(root_dir_path, pos, frame)
            if flow_full is None:
                self._status_label.setText("Could not load cellpose flow.")
                return

            nuc_t = nuclear_labels[frame]

            # Get flow/prob for this frame
            if flow_full.ndim == 4 and flow_full.shape[0] == nuclear_labels.shape[0]:
                flow_t = flow_full[frame]
            else:
                flow_t = flow_full

            if prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == nuclear_labels.shape[0]:
                prob_t = prob_full[frame]
            else:
                prob_t = prob_full

            # Run segmentation
            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=cfg.flow_scale,
                cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                flow_smoothing_sigma=cfg.flow_smoothing_sigma,
                max_iterations=cfg.max_iterations,
                uniform_growth_rate=cfg.uniform_growth_rate,
            )

            # Display in napari
            flow_mag = np.sqrt(flow_t[..., 0]**2 + flow_t[..., 1]**2)

            # Clear existing layers
            while len(self.viewer.layers) > 0:
                self.viewer.layers.pop()

            self.viewer.add_image(nuc_t, name="Nuclear Labels")
            self.viewer.add_image(prob_t, name="Cellpose Probability")
            self.viewer.add_image(flow_mag, name="Cellpose Flow Magnitude")
            self.viewer.add_labels(cell_labels, name="Cell Segmentation")

            n_cells = len(np.unique(cell_labels)) - 1
            self._status_label.setText(f"Preview complete. {n_cells} cells found.")

        except Exception as e:
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()
            self._status_label.setText(f"Preview error: {e}")

    def _on_run(self) -> None:
        root_dir = self._root_edit.text().strip()
        if not root_dir:
            self._status_label.setText("Set root directory first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Processing full stack…")

        def progress_callback(current, total):
            """Update progress bar and status label"""
            pct = int(100 * current / total)
            self._progress.setValue(pct)
            self._status_label.setText(f"Processing: {current}/{total} frames ({pct}%)")

        @thread_worker(
            connect={
                "finished": self._on_finished,
                "errored": self._on_error,
            }
        )
        def _work():
            result = run_flow_watershed(root_dir, pos, cfg, progress_callback=progress_callback)
            if result is None:
                return None
            nuc_stack, cell_stack = result
            # Save to output directory
            from ultrack_wrapper._paths import cell_segmentation_dir
            out_dir = cell_segmentation_dir(root_dir, pos)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "cell_labels.tif"
            tifffile.imwrite(
                str(out_path),
                cell_stack,
                compression="zlib",
                metadata={"axes": "TYX"},
            )
            return (nuc_stack, cell_stack, str(out_path))

        self._worker = _work()

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)

        # Get result from worker
        if self._worker and hasattr(self._worker, 'result'):
            result = self._worker.result
            if result is not None:
                nuc_stack, cell_stack, out_path = result
                # Display in napari
                self.viewer.add_labels(nuc_stack.astype(np.uint32), name="nuclei", opacity=0.3)
                self.viewer.add_labels(cell_stack.astype(np.uint32), name="cells")
                self._status_label.setText(f"Done. Saved to {Path(out_path).name}")
            else:
                self._status_label.setText("Processing failed.")
        else:
            self._status_label.setText("Done — Cell segmentation complete.")

        self._worker = None

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None

    def _on_load_results(self) -> None:
        from ultrack_wrapper._paths import cell_segmentation_dir

        root_dir = self._root_edit.text().strip()
        if not root_dir:
            self._status_label.setText("Set root directory first.")
            return

        pos = int(self._pos_spin.value())
        out_dir = cell_segmentation_dir(root_dir, pos)

        cell_path = out_dir / "cell_labels.tif"
        if not cell_path.exists():
            self._status_label.setText("No cell_labels.tif found.")
            return

        # Load stacks
        cell_stack = tifffile.imread(str(cell_path)).astype(np.uint32)

        # Also try to load nuclei for overlay
        nuc_labels = _load_nuclear_labels(root_dir, pos)

        # Display in napari
        if nuc_labels is not None:
            self.viewer.add_labels(nuc_labels.astype(np.uint32), name="nuclei", opacity=0.3)
        self.viewer.add_labels(cell_stack, name="cells")

        self._status_label.setText(f"Loaded cell_labels.tif  shape={cell_stack.shape}")

    def _on_save_params(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        cfg = self._build_config()
        Path(path).write_text(json.dumps(cfg.model_dump(), indent=2))
        self._status_label.setText(f"Parameters saved.")

    def _on_load_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        cfg = FlowWatershedConfig.from_dict(data)
        self._apply_config(cfg)
        self._status_label.setText(f"Parameters loaded.")
