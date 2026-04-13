"""Unified scrollable analysis widget: Cellpose Contours → Tracking."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.cm import get_cmap
from qtpy.QtCore import Qt, QTimer
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
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari.qt.threading import thread_worker

from ultrack_wrapper._config import CellposeContoursConfig, TrackingConfig
from ultrack_wrapper.runners.terminal import launch_in_terminal
from ultrack_wrapper.stages.s02c_cellpose_contours import (
    compute_single_from_arrays as compute_cp_contours_single,
    discover_dp_files,
    discover_prob_files,
    run as run_s02c,
)
from ultrack_wrapper.stages.s03_tracking import (
    export_ctc,
    get_labels_layer,
    get_tracks_layer,
    run as run_s03,
    run_segmentation,
    run_linking,
    run_solve,
)


# All stage outputs live directly inside the user-chosen output directory —
# no sub-folders.  foreground.tif, contours.tif, tracks.csv,
# tracked_labels.tif and the Ultrack DB all land in the same flat dir.


class UltrackAnalysisWidget(QWidget):
    """Single scrollable panel for Cellpose Contours and Tracking stages.

    All paths are derived from two shared inputs at the top:
    - **Input directory** — contains ``t*_prob.tif`` and ``t*_dp.tif`` Cellpose output files.
    - **Output directory** — base folder; outputs are written directly:
      ``foreground.tif``, ``contours.tif``, ``tracks.csv``, ``tracked_labels.tif``, ``data.db``.
    """

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

        # Background workers
        self._cp_ct_worker = None
        self._tr_worker = None
        self._seg_worker = None
        self._lnk_worker = None
        self._slv_worker = None
        self._all_worker = None

        # Cached preview data
        self._cp_ct_dp: np.ndarray | None = None
        self._cp_ct_prob: np.ndarray | None = None

        # Napari layer handles (preview)
        self._cp_ct_labels_layer = None
        self._cp_ct_contours_layer = None
        self._cp_ct_fg_layer = None

        # Debounce timers
        self._cp_ct_timer = QTimer()
        self._cp_ct_timer.setSingleShot(True)
        self._cp_ct_timer.setInterval(400)  # compute_masks is heavier
        self._cp_ct_timer.timeout.connect(self._cp_ct_update_preview)

        # ── Outer scroll area ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        self._inner_layout = QVBoxLayout()
        self._inner_layout.setAlignment(Qt.AlignTop)
        inner.setLayout(self._inner_layout)
        scroll.setWidget(inner)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        lay = self._inner_layout

        # ── Shared paths ─────────────────────────────────────────────────
        lay.addWidget(QLabel("<b>Input directory</b> (Cellpose prob maps)"))
        row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("/path/to/1a_cellpose_nucleus")
        row.addWidget(self._input_edit)
        b = QPushButton("Browse\u2026")
        b.clicked.connect(self._browse_input)
        row.addWidget(b)
        lay.addLayout(row)

        lay.addWidget(QLabel("<b>Output directory</b> (all stage outputs)"))
        row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("/path/to/experiment_output")
        row.addWidget(self._output_edit)
        b = QPushButton("Browse\u2026")
        b.clicked.connect(self._browse_output)
        row.addWidget(b)
        lay.addLayout(row)

        # ── Save / Load all parameters ───────────────────────────────────
        row = QHBoxLayout()
        b = QPushButton("Save All Parameters\u2026")
        b.clicked.connect(self._on_save_all_params)
        row.addWidget(b)
        b = QPushButton("Load All Parameters\u2026")
        b.clicked.connect(self._on_load_all_params)
        row.addWidget(b)
        lay.addLayout(row)

        # ── Build stage sections ─────────────────────────────────────────
        lay.addWidget(self._build_cp_contours_section())
        lay.addWidget(self._build_tracking_section())

        # ── Run All ──────────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_all_btn = QPushButton(
            "Run All  (Cellpose Contours \u2192 Tracking)"
        )
        self._run_all_btn.clicked.connect(self._on_run_all)
        row.addWidget(self._run_all_btn)
        self._cancel_all_btn = QPushButton("Cancel")
        self._cancel_all_btn.setEnabled(False)
        self._cancel_all_btn.clicked.connect(self._on_cancel_all)
        row.addWidget(self._cancel_all_btn)
        lay.addLayout(row)

        self._all_progress = QProgressBar()
        self._all_progress.setVisible(False)
        lay.addWidget(self._all_progress)

        self._all_status = QLabel("")
        lay.addWidget(self._all_status)

    # ══════════════════════════════════════════════════════════════════════
    # Shared path helpers
    # ══════════════════════════════════════════════════════════════════════

    def _browse_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Cellpose output directory")
        if d:
            self._input_edit.setText(d)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output base directory")
        if d:
            self._output_edit.setText(d)

    def _get_paths(self) -> tuple[str, str] | None:
        """Return (input_dir, output_dir) or set status and return None."""
        inp = self._input_edit.text().strip()
        out = self._output_edit.text().strip()
        if not inp or not out:
            return None
        return inp, out

    # ══════════════════════════════════════════════════════════════════════
    # CELLPOSE CONTOURS section (s02c)
    # ══════════════════════════════════════════════════════════════════════

    def _build_cp_contours_section(self) -> QGroupBox:
        """Build the Cellpose-native contours section (s02c)."""
        grp = QGroupBox("Contours (Cellpose)")
        lay = QVBoxLayout()

        # Preview timepoint
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview timepoint"))
        self._cp_ct_tp_idx = QSpinBox()
        self._cp_ct_tp_idx.setRange(0, 9999)
        self._cp_ct_tp_idx.setValue(0)
        self._cp_ct_tp_idx.setToolTip("Index of the timepoint to use for live preview")
        row.addWidget(self._cp_ct_tp_idx)
        lay.addLayout(row)

        # Flow threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("Flow threshold"))
        self._cp_ct_flow_thresh = QDoubleSpinBox()
        self._cp_ct_flow_thresh.setMinimum(-999999.0)
        self._cp_ct_flow_thresh.setMaximum(999999.0)
        self._cp_ct_flow_thresh.setSingleStep(0.1)
        self._cp_ct_flow_thresh.setDecimals(1)
        self._cp_ct_flow_thresh.setValue(0.4)
        self._cp_ct_flow_thresh.setToolTip(
            "Cellpose flow consistency threshold. Lower = more strict (fewer cells). Default: 0.4"
        )
        row.addWidget(self._cp_ct_flow_thresh)
        lay.addLayout(row)

        # Cell probability threshold sweep range
        row = QHBoxLayout()
        row.addWidget(QLabel("Cellprob threshold sweep"))
        self._cp_ct_cellprob_min = QDoubleSpinBox()
        self._cp_ct_cellprob_min.setMinimum(-999999.0)
        self._cp_ct_cellprob_min.setMaximum(999999.0)
        self._cp_ct_cellprob_min.setSingleStep(0.5)
        self._cp_ct_cellprob_min.setDecimals(1)
        self._cp_ct_cellprob_min.setValue(0.0)
        self._cp_ct_cellprob_min.setToolTip("Minimum cellprob threshold")
        row.addWidget(self._cp_ct_cellprob_min)
        row.addWidget(QLabel("to"))
        self._cp_ct_cellprob_max = QDoubleSpinBox()
        self._cp_ct_cellprob_max.setMinimum(-999999.0)
        self._cp_ct_cellprob_max.setMaximum(999999.0)
        self._cp_ct_cellprob_max.setSingleStep(0.5)
        self._cp_ct_cellprob_max.setDecimals(1)
        self._cp_ct_cellprob_max.setValue(0.0)
        self._cp_ct_cellprob_max.setToolTip("Maximum cellprob threshold")
        row.addWidget(self._cp_ct_cellprob_max)
        row.addWidget(QLabel("step"))
        self._cp_ct_cellprob_step = QDoubleSpinBox()
        self._cp_ct_cellprob_step.setMinimum(0.01)
        self._cp_ct_cellprob_step.setMaximum(999999.0)
        self._cp_ct_cellprob_step.setSingleStep(0.5)
        self._cp_ct_cellprob_step.setDecimals(2)
        self._cp_ct_cellprob_step.setValue(1.0)
        self._cp_ct_cellprob_step.setToolTip("Step size for cellprob threshold sweep")
        row.addWidget(self._cp_ct_cellprob_step)
        lay.addLayout(row)

        # 3D mode checkbox
        self._cp_ct_3d_chk = QCheckBox("Use 3D")
        self._cp_ct_3d_chk.setChecked(True)
        self._cp_ct_3d_chk.setToolTip("Use 3D flow-based segmentation (requires 3D dP files)")
        lay.addWidget(self._cp_ct_3d_chk)

        # Device selection
        row = QHBoxLayout()
        row.addWidget(QLabel("Device"))
        self._cp_ct_device = QComboBox()
        self._cp_ct_device.addItems(["cuda", "cpu"])
        self._cp_ct_device.setCurrentText("cuda")
        self._cp_ct_device.setToolTip("GPU (cuda) or CPU for mask computation")
        row.addWidget(self._cp_ct_device)
        lay.addLayout(row)

        # Smooth sigma
        row = QHBoxLayout()
        row.addWidget(QLabel("Smooth sigma"))
        self._cp_ct_smooth_sigma = QDoubleSpinBox()
        self._cp_ct_smooth_sigma.setMinimum(-999999.0)
        self._cp_ct_smooth_sigma.setMaximum(999999.0)
        self._cp_ct_smooth_sigma.setSingleStep(0.5)
        self._cp_ct_smooth_sigma.setDecimals(1)
        self._cp_ct_smooth_sigma.setValue(0.5)
        self._cp_ct_smooth_sigma.setToolTip("Gaussian smoothing applied to final contours map")
        row.addWidget(self._cp_ct_smooth_sigma)
        lay.addLayout(row)

        # Buttons
        row = QHBoxLayout()
        self._cp_ct_preview_btn = QPushButton("Preview")
        self._cp_ct_preview_btn.clicked.connect(self._cp_ct_on_preview)
        row.addWidget(self._cp_ct_preview_btn)
        lay.addLayout(row)

        self._cp_ct_overwrite_chk = QCheckBox("Overwrite existing files")
        lay.addWidget(self._cp_ct_overwrite_chk)

        row = QHBoxLayout()
        self._cp_ct_run_btn = QPushButton("Run Contours (Cellpose)")
        self._cp_ct_run_btn.clicked.connect(self._cp_ct_on_run)
        row.addWidget(self._cp_ct_run_btn)
        self._cp_ct_term_btn = QPushButton("Run in Terminal")
        self._cp_ct_term_btn.clicked.connect(self._cp_ct_on_run_terminal)
        row.addWidget(self._cp_ct_term_btn)
        self._cp_ct_cancel_btn = QPushButton("Cancel")
        self._cp_ct_cancel_btn.setEnabled(False)
        self._cp_ct_cancel_btn.clicked.connect(self._cp_ct_on_cancel)
        row.addWidget(self._cp_ct_cancel_btn)
        lay.addLayout(row)

        self._cp_ct_load_btn = QPushButton("Load Results")
        self._cp_ct_load_btn.clicked.connect(self._cp_ct_on_load_results)
        lay.addWidget(self._cp_ct_load_btn)

        self._cp_ct_progress = QProgressBar()
        self._cp_ct_progress.setVisible(False)
        lay.addWidget(self._cp_ct_progress)
        self._cp_ct_status = QLabel("")
        lay.addWidget(self._cp_ct_status)

        grp.setLayout(lay)
        return grp

    # ── Cellpose Contours helpers ────────────────────────────────────────

    def _cp_ct_build_config(self) -> CellposeContoursConfig:
        """Build config from current UI state."""
        # Use min threshold for single-run config
        cellprob_threshold = self._cp_ct_cellprob_min.value()

        return CellposeContoursConfig(
            cellprob_threshold=cellprob_threshold,
            do_3D=self._cp_ct_3d_chk.isChecked(),
            smooth_sigma=self._cp_ct_smooth_sigma.value(),
            device=self._cp_ct_device.currentText(),
        )

    def _cp_ct_apply_config(self, cfg: CellposeContoursConfig) -> None:
        """Apply config to UI."""
        self._cp_ct_cellprob_min.setValue(cfg.cellprob_threshold)
        self._cp_ct_3d_chk.setChecked(cfg.do_3D)
        self._cp_ct_smooth_sigma.setValue(cfg.smooth_sigma)
        self._cp_ct_device.setCurrentText(cfg.device)

    def _cp_ct_schedule(self) -> None:
        """Schedule a debounced preview update if data is cached."""
        if self._cp_ct_dp is not None and self._cp_ct_prob is not None:
            self._cp_ct_timer.start()

    def _cp_ct_on_preview(self) -> None:
        """Load dP and prob maps for a single timepoint."""
        paths = self._get_paths()
        if paths is None:
            self._cp_ct_status.setText("Set input and output directories first.")
            return
        inp, _ = paths

        dp_files = discover_dp_files(inp)
        prob_files = discover_prob_files(inp)
        tp_idx = self._cp_ct_tp_idx.value()

        if not dp_files or not prob_files:
            self._cp_ct_status.setText("No t*_dp.tif or t*_prob.tif files found.")
            return

        if len(dp_files) != len(prob_files):
            self._cp_ct_status.setText(f"Mismatch: {len(dp_files)} dp vs {len(prob_files)} prob files.")
            return

        if tp_idx >= len(dp_files):
            self._cp_ct_status.setText(f"Only {len(dp_files)} timepoints available.")
            return

        # Load single timepoint
        self._cp_ct_status.setText(f"Loading {dp_files[tp_idx].name}\u2026")
        self._cp_ct_dp = tifffile.imread(str(dp_files[tp_idx])).astype(np.float32)
        self._cp_ct_prob = tifffile.imread(str(prob_files[tp_idx])).astype(np.float32)
        self._cp_ct_update_preview()
        self._cp_ct_status.setText(f"Preview: {dp_files[tp_idx].name}")

    def _cp_ct_update_preview(self) -> None:
        """Update preview with current configuration, averaging multiple cellprob thresholds."""
        # Check if we have data
        if not (hasattr(self, '_cp_ct_dp') and self._cp_ct_dp is not None):
            return
        if not (hasattr(self, '_cp_ct_prob') and self._cp_ct_prob is not None):
            return

        dp_data = self._cp_ct_dp
        prob_data = self._cp_ct_prob

        # Generate cellprob thresholds from min/max/step
        min_thresh = self._cp_ct_cellprob_min.value()
        max_thresh = self._cp_ct_cellprob_max.value()
        step = self._cp_ct_cellprob_step.value()

        if step <= 0:
            self._cp_ct_status.setText("Error: step size must be positive")
            return

        # Generate thresholds
        if min_thresh == max_thresh:
            thresholds = [min_thresh]
        else:
            # Create range with numpy arange to avoid floating-point precision issues
            thresholds = list(np.arange(min_thresh, max_thresh + step / 2, step))

        if not thresholds:
            self._cp_ct_status.setText("Error: no cellprob thresholds generated")
            return

        self._cp_ct_status.setText(f"Computing {len(thresholds)} threshold(s): {[round(t, 2) for t in thresholds]}")

        cfg = self._cp_ct_build_config()

        # Single frame processing
        contours_list = []
        labels_list = []
        fg_list = []

        for i, thresh in enumerate(thresholds):
            cfg.cellprob_threshold = thresh
            try:
                labels, fg, contours = compute_cp_contours_single(dp_data, prob_data, cfg)
                labels_list.append(labels)
                fg_list.append(fg)
                contours_list.append(contours)
                print(f"Threshold {i+1}/{len(thresholds)}: {thresh:.2f} - {len(np.unique(labels))} unique labels")
            except Exception as e:
                self._cp_ct_status.setText(f"Error computing threshold {thresh}: {e}")
                return

        labels_stack = labels_list[0]
        fg_stack = np.mean(fg_list, axis=0)
        contours_stack = np.mean(contours_list, axis=0)

        # Update or create labels layer
        if self._cp_ct_labels_layer is None or self._cp_ct_labels_layer not in self.viewer.layers:
            self._cp_ct_labels_layer = self.viewer.add_labels(
                labels_stack.astype(np.uint32), name="cp labels preview"
            )
        else:
            self._cp_ct_labels_layer.data = labels_stack.astype(np.uint32)

        # Update or create contours layer
        if (
            self._cp_ct_contours_layer is None
            or self._cp_ct_contours_layer not in self.viewer.layers
        ):
            self._cp_ct_contours_layer = self.viewer.add_image(
                contours_stack, name="cp contours preview", colormap="hot"
            )
        else:
            self._cp_ct_contours_layer.data = contours_stack

        # Update or create foreground layer
        if self._cp_ct_fg_layer is None or self._cp_ct_fg_layer not in self.viewer.layers:
            self._cp_ct_fg_layer = self.viewer.add_image(
                fg_stack, name="cp foreground preview", colormap="green", visible=False
            )
        else:
            self._cp_ct_fg_layer.data = fg_stack

        threshold_count = len(thresholds)
        self._cp_ct_status.setText(f"Preview: 1 frame × {threshold_count} threshold(s) averaged")

    def _cp_ct_run_sweep_and_average(self, inp: str, out: str, overwrite: bool):
        """Generator that sweeps cellprob thresholds and averages results."""
        from pathlib import Path

        dp_files = discover_dp_files(inp)
        prob_files = discover_prob_files(inp)

        if not dp_files or not prob_files or len(dp_files) != len(prob_files):
            yield (0, 0, "No valid dP/prob file pairs found")
            return

        # Generate cellprob thresholds from min/max/step
        min_thresh = self._cp_ct_cellprob_min.value()
        max_thresh = self._cp_ct_cellprob_max.value()
        step = self._cp_ct_cellprob_step.value()

        if step <= 0:
            yield (0, 0, "Error: step size must be positive")
            return

        if min_thresh == max_thresh:
            thresholds = [min_thresh]
        else:
            thresholds = list(np.arange(min_thresh, max_thresh + step / 2, step))

        if not thresholds:
            yield (0, 0, "Error: no cellprob thresholds generated")
            return

        total_frames = len(dp_files)
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)

        fg_path = out_path / "foreground.tif"
        ct_path = out_path / "contours.tif"

        if not overwrite and fg_path.exists() and ct_path.exists():
            yield (0, 1, "foreground.tif and contours.tif already exist, skipping")
            return

        cfg = self._cp_ct_build_config()
        fg_frames = []
        ct_frames = []

        yield (0, total_frames, f"Processing {total_frames} frame(s) with {len(thresholds)} threshold(s)")

        for frame_idx, (dp_file, prob_file) in enumerate(zip(dp_files, prob_files)):
            try:
                dp = tifffile.imread(str(dp_file)).astype(np.float32)
                prob = tifffile.imread(str(prob_file)).astype(np.float32)

                fg_list = []
                ct_list = []

                for thresh in thresholds:
                    cfg.cellprob_threshold = thresh
                    _, fg, ct = compute_cp_contours_single(dp, prob, cfg)
                    fg_list.append(fg)
                    ct_list.append(ct)

                # Average across thresholds
                fg_avg = np.mean(fg_list, axis=0).astype(np.float32)
                ct_avg = np.mean(ct_list, axis=0).astype(np.float32)

                fg_frames.append(fg_avg)
                ct_frames.append(ct_avg)

                t_str = dp_file.name.split("_")[0]
                yield (frame_idx + 1, total_frames, t_str)
            except Exception as e:
                yield (frame_idx + 1, total_frames, f"Error {frame_idx}: {e}")
                raise

        # Write stacks
        fg_stack = np.stack(fg_frames, axis=0)
        ct_stack = np.stack(ct_frames, axis=0)
        tifffile.imwrite(str(fg_path), fg_stack, compression="zlib")
        tifffile.imwrite(str(ct_path), ct_stack, compression="zlib")

        yield (total_frames, total_frames, "Done")

    def _cp_ct_on_run(self) -> None:
        """Run cellpose contours stage with threshold sweep and averaging."""
        paths = self._get_paths()
        if paths is None:
            self._cp_ct_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        overwrite = self._cp_ct_overwrite_chk.isChecked()
        self._cp_ct_run_btn.setEnabled(False)
        self._cp_ct_cancel_btn.setEnabled(True)
        self._cp_ct_progress.setVisible(True)
        self._cp_ct_status.setText("Starting\u2026")

        @thread_worker(
            connect={
                "yielded": self._cp_ct_on_progress,
                "finished": self._cp_ct_on_finished,
                "errored": self._cp_ct_on_error,
            }
        )
        def _work():
            for u in self._cp_ct_run_sweep_and_average(inp, out, overwrite):
                yield u

        self._cp_ct_worker = _work()
        self._cp_ct_worker.aborted.connect(self._cp_ct_on_cancelled)

    def _cp_ct_on_run_terminal(self) -> None:
        """Launch cellpose contours in terminal."""
        paths = self._get_paths()
        if paths is None:
            self._cp_ct_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        cfg = self._cp_ct_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_cp_ct_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._cp_ct_overwrite_chk.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s02c_cellpose_contours"
            f" --input-dir \"{inp}\""
            f" --output-dir \"{out}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._cp_ct_status.setText("Launched cellpose contours in terminal.")
        except Exception as e:
            self._cp_ct_status.setText(f"Terminal error: {e}")

    def _cp_ct_on_progress(self, u: tuple) -> None:
        """Update progress bar during run."""
        done, total, label = u
        self._cp_ct_progress.setMaximum(max(total, 1))
        self._cp_ct_progress.setValue(done)
        self._cp_ct_status.setText(f"[{done}/{total}] {label}")

    def _cp_ct_on_cancel(self) -> None:
        """Cancel the run."""
        if self._cp_ct_worker is not None:
            self._cp_ct_worker.quit()

    def _cp_ct_on_cancelled(self) -> None:
        """Handle cancelled state."""
        self._cp_ct_run_btn.setEnabled(True)
        self._cp_ct_cancel_btn.setEnabled(False)
        self._cp_ct_progress.setVisible(False)
        self._cp_ct_worker = None
        self._cp_ct_status.setText("Cancelled.")

    def _cp_ct_on_finished(self) -> None:
        """Handle successful completion."""
        self._cp_ct_run_btn.setEnabled(True)
        self._cp_ct_cancel_btn.setEnabled(False)
        self._cp_ct_progress.setVisible(False)
        self._cp_ct_worker = None
        self._cp_ct_status.setText("Done \u2014 foreground.tif and contours.tif written.")
        self._cp_ct_load_stack()

    def _cp_ct_on_error(self, exc: Exception) -> None:
        """Handle error."""
        self._cp_ct_run_btn.setEnabled(True)
        self._cp_ct_cancel_btn.setEnabled(False)
        self._cp_ct_progress.setVisible(False)
        self._cp_ct_worker = None
        self._cp_ct_status.setText(f"Error: {exc}")

    def _cp_ct_load_stack(self) -> None:
        """Load output files from disk."""
        paths = self._get_paths()
        if paths is None:
            return
        fg_path = Path(paths[1]) / "foreground.tif"
        ct_path = Path(paths[1]) / "contours.tif"
        if not fg_path.exists():
            self._cp_ct_status.setText("foreground.tif not found.")
            return
        if not ct_path.exists():
            self._cp_ct_status.setText("contours.tif not found.")
            return
        fg_stack = tifffile.imread(str(fg_path))
        ct_stack = tifffile.imread(str(ct_path))
        self.viewer.add_image(fg_stack, name="cp foreground", colormap="green")
        self.viewer.add_image(ct_stack, name="cp contours", colormap="hot")
        self._cp_ct_status.setText(f"Loaded foreground + contours: {fg_stack.shape}")

    def _cp_ct_on_load_results(self) -> None:
        """Load results button handler."""
        self._cp_ct_load_stack()

    # ══════════════════════════════════════════════════════════════════════
    # TRACKING section
    # ══════════════════════════════════════════════════════════════════════

    def _build_tracking_section(self) -> QGroupBox:
        grp = QGroupBox("Tracking")
        lay = QVBoxLayout()

        # ── SEGMENTATION sub-section ─────────────────────────────────────
        seg = QGroupBox("Segmentation hypotheses")
        sl = QVBoxLayout()

        # Overwrite checkbox for segmentation
        row = QHBoxLayout()
        self._tr_overwrite_seg_chk = QCheckBox("Overwrite")
        self._tr_overwrite_seg_chk.setChecked(True)
        self._tr_overwrite_seg_chk.setToolTip("Re-run segmentation even if candidates already exist in the database")
        row.addWidget(self._tr_overwrite_seg_chk)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min area"))
        self._tr_min_area = QSpinBox()
        self._tr_min_area.setRange(1, 10_000_000)
        self._tr_min_area.setValue(100)
        self._tr_min_area.setToolTip("Minimum area (pixels) for a segmentation hypothesis; smaller candidates are discarded")
        row.addWidget(self._tr_min_area)
        row.addWidget(QLabel("Max area"))
        self._tr_max_area = QSpinBox()
        self._tr_max_area.setRange(1, 10_000_000)
        self._tr_max_area.setValue(1_000_000)
        self._tr_max_area.setToolTip("Maximum area (pixels) for a segmentation hypothesis; larger regions are discarded")
        row.addWidget(self._tr_max_area)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min frontier"))
        self._tr_min_front = QDoubleSpinBox()
        self._tr_min_front.setRange(0.0, 1.0)
        self._tr_min_front.setSingleStep(0.05)
        self._tr_min_front.setDecimals(3)
        self._tr_min_front.setToolTip("Minimum contour/frontier value required to accept a candidate")
        row.addWidget(self._tr_min_front)
        row.addWidget(QLabel("FG threshold"))
        self._tr_seg_thresh = QDoubleSpinBox()
        self._tr_seg_thresh.setRange(0.0, 1.0)
        self._tr_seg_thresh.setSingleStep(0.05)
        self._tr_seg_thresh.setDecimals(2)
        self._tr_seg_thresh.setValue(0.5)
        self._tr_seg_thresh.setToolTip("Foreground probability threshold for extracting segmentation candidates")
        row.addWidget(self._tr_seg_thresh)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("WS hierarchy"))
        self._tr_ws_combo = QComboBox()
        self._tr_ws_combo.addItems(["area", "dynamics", "volume"])
        self._tr_ws_combo.setToolTip("Criterion for ordering watershed merges: \"area\", \"dynamics\", or \"volume\"")
        row.addWidget(self._tr_ws_combo)
        row.addWidget(QLabel("Aniso. pen."))
        self._tr_aniso = QDoubleSpinBox()
        self._tr_aniso.setRange(-10.0, 10.0)
        self._tr_aniso.setSingleStep(0.1)
        self._tr_aniso.setDecimals(1)
        self._tr_aniso.setToolTip("Penalty for inter-frame links to compensate for anisotropic voxel spacing; 0 = no penalty")
        row.addWidget(self._tr_aniso)
        sl.addLayout(row)

        # Workers for segmentation
        row = QHBoxLayout()
        row.addWidget(QLabel("Workers"))
        self._tr_seg_workers = QSpinBox()
        self._tr_seg_workers.setRange(1, 64)
        self._tr_seg_workers.setValue(1)
        self._tr_seg_workers.setToolTip("Number of parallel workers for the segmentation stage")
        row.addWidget(self._tr_seg_workers)
        sl.addLayout(row)

        # Buttons for segmentation
        row = QHBoxLayout()
        self._tr_seg_run_btn = QPushButton("Run Segmentation")
        self._tr_seg_run_btn.clicked.connect(self._tr_on_run_segmentation)
        row.addWidget(self._tr_seg_run_btn)
        self._tr_seg_term_btn = QPushButton("Run in Terminal")
        self._tr_seg_term_btn.clicked.connect(self._tr_on_run_seg_terminal)
        row.addWidget(self._tr_seg_term_btn)
        self._tr_seg_cancel_btn = QPushButton("Cancel")
        self._tr_seg_cancel_btn.setEnabled(False)
        self._tr_seg_cancel_btn.clicked.connect(self._tr_on_seg_cancel)
        row.addWidget(self._tr_seg_cancel_btn)
        sl.addLayout(row)

        self._tr_seg_progress = QProgressBar()
        self._tr_seg_progress.setVisible(False)
        sl.addWidget(self._tr_seg_progress)
        self._tr_seg_status = QLabel("")
        sl.addWidget(self._tr_seg_status)

        seg.setLayout(sl)
        lay.addWidget(seg)

        # ── LINKING sub-section ──────────────────────────────────────────
        lnk = QGroupBox("Linking")
        ll = QVBoxLayout()

        # Overwrite checkbox for linking
        row = QHBoxLayout()
        self._tr_overwrite_lnk_chk = QCheckBox("Overwrite")
        self._tr_overwrite_lnk_chk.setChecked(True)
        self._tr_overwrite_lnk_chk.setToolTip("Re-run linking even if the link graph already exists in the database")
        row.addWidget(self._tr_overwrite_lnk_chk)
        ll.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max distance"))
        self._tr_max_dist = QDoubleSpinBox()
        self._tr_max_dist.setRange(0.1, 500.0)
        self._tr_max_dist.setSingleStep(1.0)
        self._tr_max_dist.setDecimals(1)
        self._tr_max_dist.setValue(15.0)
        self._tr_max_dist.setToolTip("Maximum centroid-to-centroid distance (pixels) for linking a candidate across frames")
        row.addWidget(self._tr_max_dist)
        row.addWidget(QLabel("Max neighbors"))
        self._tr_max_nb = QSpinBox()
        self._tr_max_nb.setRange(1, 50)
        self._tr_max_nb.setValue(5)
        self._tr_max_nb.setToolTip("Maximum number of candidate links considered per segment per frame")
        row.addWidget(self._tr_max_nb)
        ll.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Distance weight"))
        self._tr_dist_w = QDoubleSpinBox()
        self._tr_dist_w.setRange(0.0, 10.0)
        self._tr_dist_w.setSingleStep(0.1)
        self._tr_dist_w.setDecimals(2)
        self._tr_dist_w.setToolTip("Weight of the distance term in the link-cost function; 0 = pure overlap-based linking")
        row.addWidget(self._tr_dist_w)
        ll.addLayout(row)

        # Workers for linking
        row = QHBoxLayout()
        row.addWidget(QLabel("Workers"))
        self._tr_lnk_workers = QSpinBox()
        self._tr_lnk_workers.setRange(1, 64)
        self._tr_lnk_workers.setValue(1)
        self._tr_lnk_workers.setToolTip("Number of parallel workers for the linking stage")
        row.addWidget(self._tr_lnk_workers)
        ll.addLayout(row)

        # Buttons for linking
        row = QHBoxLayout()
        self._tr_lnk_run_btn = QPushButton("Run Linking")
        self._tr_lnk_run_btn.clicked.connect(self._tr_on_run_linking)
        row.addWidget(self._tr_lnk_run_btn)
        self._tr_lnk_term_btn = QPushButton("Run in Terminal")
        self._tr_lnk_term_btn.clicked.connect(self._tr_on_run_lnk_terminal)
        row.addWidget(self._tr_lnk_term_btn)
        self._tr_lnk_cancel_btn = QPushButton("Cancel")
        self._tr_lnk_cancel_btn.setEnabled(False)
        self._tr_lnk_cancel_btn.clicked.connect(self._tr_on_lnk_cancel)
        row.addWidget(self._tr_lnk_cancel_btn)
        ll.addLayout(row)

        self._tr_lnk_progress = QProgressBar()
        self._tr_lnk_progress.setVisible(False)
        ll.addWidget(self._tr_lnk_progress)
        self._tr_lnk_status = QLabel("")
        ll.addWidget(self._tr_lnk_status)

        lnk.setLayout(ll)
        lay.addWidget(lnk)

        # ── SOLVER (ILP) sub-section ─────────────────────────────────────
        slv = QGroupBox("Solver (ILP)")
        sv = QVBoxLayout()

        # Overwrite checkbox for solve
        row = QHBoxLayout()
        self._tr_overwrite_slv_chk = QCheckBox("Overwrite")
        self._tr_overwrite_slv_chk.setChecked(True)
        self._tr_overwrite_slv_chk.setToolTip("Re-run the ILP solver even if a solution already exists in the database")
        row.addWidget(self._tr_overwrite_slv_chk)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Appear"))
        self._tr_appear = QDoubleSpinBox()
        self._tr_appear.setRange(-100.0, 0.0)
        self._tr_appear.setSingleStep(0.001)
        self._tr_appear.setDecimals(4)
        self._tr_appear.setValue(-0.001)
        self._tr_appear.setToolTip("ILP cost for a track appearing mid-sequence; more negative = fewer track starts")
        row.addWidget(self._tr_appear)
        row.addWidget(QLabel("Disappear"))
        self._tr_disappear = QDoubleSpinBox()
        self._tr_disappear.setRange(-100.0, 0.0)
        self._tr_disappear.setSingleStep(0.001)
        self._tr_disappear.setDecimals(4)
        self._tr_disappear.setValue(-0.001)
        self._tr_disappear.setToolTip("ILP cost for a track ending mid-sequence; more negative = fewer track ends")
        row.addWidget(self._tr_disappear)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Division"))
        self._tr_division = QDoubleSpinBox()
        self._tr_division.setRange(-100.0, 0.0)
        self._tr_division.setSingleStep(0.001)
        self._tr_division.setDecimals(4)
        self._tr_division.setValue(-0.001)
        self._tr_division.setToolTip("ILP cost for a cell division event; more negative = more divisions detected")
        row.addWidget(self._tr_division)
        row.addWidget(QLabel("Link func"))
        self._tr_link_func = QComboBox()
        self._tr_link_func.addItems(["power", "identity"])
        self._tr_link_func.setToolTip("Transform link scores: \"power\" raises to the exponent; \"identity\" uses scores directly")
        row.addWidget(self._tr_link_func)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Power"))
        self._tr_power = QDoubleSpinBox()
        self._tr_power.setRange(0.1, 20.0)
        self._tr_power.setSingleStep(0.5)
        self._tr_power.setDecimals(1)
        self._tr_power.setValue(4.0)
        self._tr_power.setToolTip("Exponent for the \"power\" link function; higher values amplify strong-vs-weak link differences")
        row.addWidget(self._tr_power)
        row.addWidget(QLabel("Bias"))
        self._tr_bias = QDoubleSpinBox()
        self._tr_bias.setRange(-10.0, 0.0)
        self._tr_bias.setSingleStep(0.01)
        self._tr_bias.setDecimals(3)
        self._tr_bias.setToolTip("Constant added to all link scores; negative bias discourages linking overall")
        row.addWidget(self._tr_bias)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gap"))
        self._tr_gap = QDoubleSpinBox()
        self._tr_gap.setRange(0.0, 1.0)
        self._tr_gap.setSingleStep(0.001)
        self._tr_gap.setDecimals(4)
        self._tr_gap.setValue(0.001)
        self._tr_gap.setToolTip("Relative optimality gap tolerance for the ILP solver; smaller = tighter solution, slower")
        row.addWidget(self._tr_gap)
        row.addWidget(QLabel("Time limit (s)"))
        self._tr_time_limit = QSpinBox()
        self._tr_time_limit.setRange(10, 360_000)
        self._tr_time_limit.setValue(36_000)
        self._tr_time_limit.setToolTip("Maximum wall-clock time (seconds) the ILP solver may run")
        row.addWidget(self._tr_time_limit)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Window size (0=all)"))
        self._tr_window = QSpinBox()
        self._tr_window.setRange(0, 10_000)
        self._tr_window.setToolTip("Temporal window size for the ILP; 0 = solve the entire sequence at once")
        row.addWidget(self._tr_window)
        sv.addLayout(row)

        # Buttons for solve
        row = QHBoxLayout()
        self._tr_slv_run_btn = QPushButton("Run Solve")
        self._tr_slv_run_btn.clicked.connect(self._tr_on_run_solve)
        row.addWidget(self._tr_slv_run_btn)
        self._tr_slv_term_btn = QPushButton("Run in Terminal")
        self._tr_slv_term_btn.clicked.connect(self._tr_on_run_slv_terminal)
        row.addWidget(self._tr_slv_term_btn)
        self._tr_slv_cancel_btn = QPushButton("Cancel")
        self._tr_slv_cancel_btn.setEnabled(False)
        self._tr_slv_cancel_btn.clicked.connect(self._tr_on_slv_cancel)
        row.addWidget(self._tr_slv_cancel_btn)
        sv.addLayout(row)

        self._tr_slv_progress = QProgressBar()
        self._tr_slv_progress.setVisible(False)
        sv.addWidget(self._tr_slv_progress)
        self._tr_slv_status = QLabel("")
        sv.addWidget(self._tr_slv_status)

        slv.setLayout(sv)
        lay.addWidget(slv)

        # ── Full pipeline buttons ────────────────────────────────────────
        row = QHBoxLayout()
        self._tr_run_btn = QPushButton("Run Full Tracking Pipeline")
        self._tr_run_btn.clicked.connect(self._tr_on_run)
        row.addWidget(self._tr_run_btn)
        self._tr_term_btn = QPushButton("Run in Terminal")
        self._tr_term_btn.clicked.connect(self._tr_on_run_terminal)
        row.addWidget(self._tr_term_btn)
        self._tr_cancel_btn = QPushButton("Cancel")
        self._tr_cancel_btn.setEnabled(False)
        self._tr_cancel_btn.clicked.connect(self._tr_on_cancel)
        row.addWidget(self._tr_cancel_btn)
        lay.addLayout(row)

        self._tr_load_btn = QPushButton("Load Results into Viewer")
        self._tr_load_btn.clicked.connect(self._tr_on_load_results)
        lay.addWidget(self._tr_load_btn)

        self._tr_export_ctc_btn = QPushButton("Export CTC\u2026")
        self._tr_export_ctc_btn.clicked.connect(self._tr_on_export_ctc)
        lay.addWidget(self._tr_export_ctc_btn)

        self._tr_progress = QProgressBar()
        self._tr_progress.setVisible(False)
        lay.addWidget(self._tr_progress)
        self._tr_status = QLabel("")
        lay.addWidget(self._tr_status)

        # ── Inspect DB section ───────────────────────────────────────────
        db_grp = QGroupBox("Inspect Database")
        db_lay = QVBoxLayout()

        row = QHBoxLayout()
        self._db_load_candidates_btn = QPushButton("Load Candidates")
        self._db_load_candidates_btn.clicked.connect(self._db_on_load_candidates)
        row.addWidget(self._db_load_candidates_btn)
        self._db_load_links_btn = QPushButton("Load Links")
        self._db_load_links_btn.clicked.connect(self._db_on_load_links)
        row.addWidget(self._db_load_links_btn)
        self._db_load_divisions_btn = QPushButton("Load Divisions")
        self._db_load_divisions_btn.clicked.connect(self._db_on_load_divisions)
        row.addWidget(self._db_load_divisions_btn)
        self._db_load_labels_btn = QPushButton("Load Labels")
        self._db_load_labels_btn.clicked.connect(self._db_on_load_labels)
        row.addWidget(self._db_load_labels_btn)
        self._db_show_stats_btn = QPushButton("Show Stats")
        self._db_show_stats_btn.clicked.connect(self._db_on_show_stats)
        row.addWidget(self._db_show_stats_btn)
        db_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Size candidates by:"))
        self._db_size_by_combo = QComboBox()
        self._db_size_by_combo.addItems(["none", "area", "node_prob", "frontier", "height"])
        self._db_size_by_combo.setEnabled(False)
        self._db_size_by_combo.currentTextChanged.connect(self._db_on_size_by_changed)
        row.addWidget(self._db_size_by_combo)

        row.addWidget(QLabel("Scale:"))
        self._db_size_scale = QDoubleSpinBox()
        self._db_size_scale.setRange(1.0, 100.0)
        self._db_size_scale.setValue(5.0)
        self._db_size_scale.setSingleStep(1.0)
        self._db_size_scale.setEnabled(False)
        self._db_size_scale.valueChanged.connect(self._db_on_size_scale_changed)
        row.addWidget(self._db_size_scale)
        row.addStretch()
        db_lay.addLayout(row)

        self._db_status = QTextEdit()
        self._db_status.setReadOnly(True)
        self._db_status.setMaximumHeight(80)
        db_lay.addWidget(self._db_status)

        db_grp.setLayout(db_lay)
        lay.addWidget(db_grp)

        # Store dataframe for colour mapping
        self._db_candidates_df = None

        grp.setLayout(lay)
        return grp

    # ── Tracking helpers ──────────────────────────────────────────────────

    def _tr_build_config(self) -> TrackingConfig:
        return TrackingConfig(
            min_area=self._tr_min_area.value(),
            max_area=self._tr_max_area.value(),
            min_frontier=self._tr_min_front.value(),
            threshold=self._tr_seg_thresh.value(),
            ws_hierarchy=self._tr_ws_combo.currentText(),
            anisotropy_penalization=self._tr_aniso.value(),
            n_workers=self._tr_seg_workers.value(),
            max_distance=self._tr_max_dist.value(),
            max_neighbors=self._tr_max_nb.value(),
            distance_weight=self._tr_dist_w.value(),
            appear_weight=self._tr_appear.value(),
            disappear_weight=self._tr_disappear.value(),
            division_weight=self._tr_division.value(),
            link_function=self._tr_link_func.currentText(),
            power=self._tr_power.value(),
            bias=self._tr_bias.value(),
            solution_gap=self._tr_gap.value(),
            time_limit=self._tr_time_limit.value(),
            window_size=self._tr_window.value(),
            overwrite_segmentation=self._tr_overwrite_seg_chk.isChecked(),
            overwrite_linking=self._tr_overwrite_lnk_chk.isChecked(),
            overwrite_solve=self._tr_overwrite_slv_chk.isChecked(),
        )

    def _tr_apply_config(self, cfg: TrackingConfig) -> None:
        self._tr_min_area.setValue(cfg.min_area)
        self._tr_max_area.setValue(cfg.max_area)
        self._tr_min_front.setValue(cfg.min_frontier)
        self._tr_seg_thresh.setValue(cfg.threshold)
        self._tr_ws_combo.setCurrentText(cfg.ws_hierarchy)
        self._tr_aniso.setValue(cfg.anisotropy_penalization)
        self._tr_seg_workers.setValue(cfg.n_workers)
        self._tr_max_dist.setValue(cfg.max_distance)
        self._tr_max_nb.setValue(cfg.max_neighbors)
        self._tr_dist_w.setValue(cfg.distance_weight)
        self._tr_appear.setValue(cfg.appear_weight)
        self._tr_disappear.setValue(cfg.disappear_weight)
        self._tr_division.setValue(cfg.division_weight)
        self._tr_link_func.setCurrentText(cfg.link_function)
        self._tr_power.setValue(cfg.power)
        self._tr_bias.setValue(cfg.bias)
        self._tr_gap.setValue(cfg.solution_gap)
        self._tr_time_limit.setValue(cfg.time_limit)
        self._tr_window.setValue(cfg.window_size)
        self._tr_overwrite_seg_chk.setChecked(cfg.overwrite_segmentation)
        self._tr_overwrite_lnk_chk.setChecked(cfg.overwrite_linking)
        self._tr_overwrite_slv_chk.setChecked(cfg.overwrite_solve)

    def _tr_on_run(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_status.setText("Set input and output directories first.")
            return
        _, out = paths
        fg_path = str(Path(out) / "foreground.tif")
        ct_path = str(Path(out) / "contours.tif")
        wd = out
        cfg = self._tr_build_config()
        self._tr_run_btn.setEnabled(False)
        self._tr_cancel_btn.setEnabled(True)
        self._tr_progress.setVisible(True)
        self._tr_status.setText("Starting\u2026")

        @thread_worker(connect={
            "yielded": self._tr_on_progress,
            "finished": self._tr_on_finished,
            "errored": self._tr_on_error,
        })
        def _work():
            for u in run_s03(fg_path, ct_path, wd, cfg):
                yield u

        self._tr_worker = _work()
        self._tr_worker.aborted.connect(self._tr_on_cancelled)

    def _tr_on_run_terminal(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_status.setText("Set input and output directories first.")
            return
        _, out = paths
        fg_path = str(Path(out) / "foreground.tif")
        ct_path = str(Path(out) / "contours.tif")
        wd = out
        cfg = self._tr_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_tr_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        cmd = (
            f"python -m ultrack_wrapper.stages.s03_tracking"
            f" --foreground \"{fg_path}\""
            f" --contours \"{ct_path}\""
            f" --working-dir \"{wd}\""
            f" --config \"{cfg_path}\""
        )
        try:
            launch_in_terminal(cmd)
            self._tr_status.setText("Launched tracking in terminal.")
        except Exception as e:
            self._tr_status.setText(f"Terminal error: {e}")

    def _tr_on_run_seg_terminal(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_seg_status.setText("Set input and output directories first.")
            return
        _, out = paths
        fg_path = str(Path(out) / "foreground.tif")
        ct_path = str(Path(out) / "contours.tif")
        wd = out
        cfg = self._tr_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_tr_seg_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        cmd = (
            f"python -m ultrack_wrapper.stages.s03_tracking --stage segmentation"
            f" --foreground \"{fg_path}\""
            f" --contours \"{ct_path}\""
            f" --working-dir \"{wd}\""
            f" --config \"{cfg_path}\""
        )
        try:
            launch_in_terminal(cmd)
            self._tr_seg_status.setText("Launched segmentation in terminal.")
        except Exception as e:
            self._tr_seg_status.setText(f"Terminal error: {e}")

    def _tr_on_run_lnk_terminal(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_lnk_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = out
        cfg = self._tr_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_tr_lnk_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        cmd = (
            f"python -m ultrack_wrapper.stages.s03_tracking --stage linking"
            f" --working-dir \"{wd}\""
            f" --config \"{cfg_path}\""
        )
        try:
            launch_in_terminal(cmd)
            self._tr_lnk_status.setText("Launched linking in terminal.")
        except Exception as e:
            self._tr_lnk_status.setText(f"Terminal error: {e}")

    def _tr_on_run_slv_terminal(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_slv_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = out
        cfg = self._tr_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_tr_slv_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        cmd = (
            f"python -m ultrack_wrapper.stages.s03_tracking --stage solve"
            f" --working-dir \"{wd}\""
            f" --config \"{cfg_path}\""
        )
        try:
            launch_in_terminal(cmd)
            self._tr_slv_status.setText("Launched solve in terminal.")
        except Exception as e:
            self._tr_slv_status.setText(f"Terminal error: {e}")

    def _tr_on_progress(self, u: tuple) -> None:
        done, total, label = u
        self._tr_progress.setMaximum(max(total, 1))
        self._tr_progress.setValue(done)
        self._tr_status.setText(label)

    def _tr_on_cancel(self) -> None:
        if self._tr_worker is not None:
            self._tr_worker.quit()

    def _tr_on_cancelled(self) -> None:
        self._tr_run_btn.setEnabled(True)
        self._tr_cancel_btn.setEnabled(False)
        self._tr_progress.setVisible(False)
        self._tr_worker = None
        self._tr_status.setText("Cancelled.")

    def _tr_on_finished(self) -> None:
        self._tr_run_btn.setEnabled(True)
        self._tr_cancel_btn.setEnabled(False)
        self._tr_progress.setVisible(False)
        self._tr_worker = None
        self._tr_status.setText("Tracking complete \u2014 loading results\u2026")
        self._tr_load_results()

    def _tr_on_error(self, exc: Exception) -> None:
        self._tr_run_btn.setEnabled(True)
        self._tr_cancel_btn.setEnabled(False)
        self._tr_progress.setVisible(False)
        self._tr_worker = None
        self._tr_status.setText(f"Error: {exc}")

    # ── Per-stage run methods ────────────────────────────────────────────

    def _tr_on_run_segmentation(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_seg_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        fg_path = str(Path(out) / "foreground.tif")
        ct_path = str(Path(out) / "contours.tif")
        wd = out
        cfg = self._tr_build_config()
        self._tr_seg_run_btn.setEnabled(False)
        self._tr_seg_cancel_btn.setEnabled(True)
        self._tr_seg_progress.setVisible(True)
        self._tr_seg_status.setText("Starting\u2026")

        @thread_worker(connect={
            "yielded": self._tr_seg_on_progress,
            "finished": self._tr_seg_on_finished,
            "errored": self._tr_seg_on_error,
        })
        def _work():
            for u in run_segmentation(fg_path, ct_path, wd, cfg, overwrite=cfg.overwrite_segmentation):
                yield u

        self._seg_worker = _work()
        self._seg_worker.aborted.connect(self._tr_seg_on_cancelled)

    def _tr_seg_on_progress(self, u: tuple) -> None:
        done, total, label = u
        self._tr_seg_progress.setMaximum(max(total, 1))
        self._tr_seg_progress.setValue(done)
        self._tr_seg_status.setText(label)

    def _tr_on_seg_cancel(self) -> None:
        if self._seg_worker is not None:
            self._seg_worker.quit()

    def _tr_seg_on_cancelled(self) -> None:
        self._tr_seg_run_btn.setEnabled(True)
        self._tr_seg_cancel_btn.setEnabled(False)
        self._tr_seg_progress.setVisible(False)
        self._seg_worker = None
        self._tr_seg_status.setText("Cancelled.")

    def _tr_seg_on_finished(self) -> None:
        self._tr_seg_run_btn.setEnabled(True)
        self._tr_seg_cancel_btn.setEnabled(False)
        self._tr_seg_progress.setVisible(False)
        self._seg_worker = None
        self._tr_seg_status.setText("Segmentation complete.")

    def _tr_seg_on_error(self, exc: Exception) -> None:
        self._tr_seg_run_btn.setEnabled(True)
        self._tr_seg_cancel_btn.setEnabled(False)
        self._tr_seg_progress.setVisible(False)
        self._seg_worker = None
        self._tr_seg_status.setText(f"Error: {exc}")

    def _tr_on_run_linking(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_lnk_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = out
        cfg = self._tr_build_config()
        self._tr_lnk_run_btn.setEnabled(False)
        self._tr_lnk_cancel_btn.setEnabled(True)
        self._tr_lnk_progress.setVisible(True)
        self._tr_lnk_status.setText("Starting\u2026")

        @thread_worker(connect={
            "yielded": self._tr_lnk_on_progress,
            "finished": self._tr_lnk_on_finished,
            "errored": self._tr_lnk_on_error,
        })
        def _work():
            for u in run_linking(wd, cfg, overwrite=cfg.overwrite_linking):
                yield u

        self._lnk_worker = _work()
        self._lnk_worker.aborted.connect(self._tr_lnk_on_cancelled)

    def _tr_lnk_on_progress(self, u: tuple) -> None:
        done, total, label = u
        self._tr_lnk_progress.setMaximum(max(total, 1))
        self._tr_lnk_progress.setValue(done)
        self._tr_lnk_status.setText(label)

    def _tr_on_lnk_cancel(self) -> None:
        if self._lnk_worker is not None:
            self._lnk_worker.quit()

    def _tr_lnk_on_cancelled(self) -> None:
        self._tr_lnk_run_btn.setEnabled(True)
        self._tr_lnk_cancel_btn.setEnabled(False)
        self._tr_lnk_progress.setVisible(False)
        self._lnk_worker = None
        self._tr_lnk_status.setText("Cancelled.")

    def _tr_lnk_on_finished(self) -> None:
        self._tr_lnk_run_btn.setEnabled(True)
        self._tr_lnk_cancel_btn.setEnabled(False)
        self._tr_lnk_progress.setVisible(False)
        self._lnk_worker = None
        self._tr_lnk_status.setText("Linking complete.")

    def _tr_lnk_on_error(self, exc: Exception) -> None:
        self._tr_lnk_run_btn.setEnabled(True)
        self._tr_lnk_cancel_btn.setEnabled(False)
        self._tr_lnk_progress.setVisible(False)
        self._lnk_worker = None
        self._tr_lnk_status.setText(f"Error: {exc}")

    def _tr_on_run_solve(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_slv_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = out
        cfg = self._tr_build_config()
        self._tr_slv_run_btn.setEnabled(False)
        self._tr_slv_cancel_btn.setEnabled(True)
        self._tr_slv_progress.setVisible(True)
        self._tr_slv_status.setText("Starting\u2026")

        @thread_worker(connect={
            "yielded": self._tr_slv_on_progress,
            "finished": self._tr_slv_on_finished,
            "errored": self._tr_slv_on_error,
        })
        def _work():
            for u in run_solve(wd, cfg, overwrite=cfg.overwrite_solve):
                yield u

        self._slv_worker = _work()
        self._slv_worker.aborted.connect(self._tr_slv_on_cancelled)

    def _tr_slv_on_progress(self, u: tuple) -> None:
        done, total, label = u
        self._tr_slv_progress.setMaximum(max(total, 1))
        self._tr_slv_progress.setValue(done)
        self._tr_slv_status.setText(label)

    def _tr_on_slv_cancel(self) -> None:
        if self._slv_worker is not None:
            self._slv_worker.quit()

    def _tr_slv_on_cancelled(self) -> None:
        self._tr_slv_run_btn.setEnabled(True)
        self._tr_slv_cancel_btn.setEnabled(False)
        self._tr_slv_progress.setVisible(False)
        self._slv_worker = None
        self._tr_slv_status.setText("Cancelled.")

    def _tr_slv_on_finished(self) -> None:
        self._tr_slv_run_btn.setEnabled(True)
        self._tr_slv_cancel_btn.setEnabled(False)
        self._tr_slv_progress.setVisible(False)
        self._slv_worker = None
        self._tr_slv_status.setText("Solve complete \u2014 loading results\u2026")
        self._tr_load_results()

    def _tr_slv_on_error(self, exc: Exception) -> None:
        self._tr_slv_run_btn.setEnabled(True)
        self._tr_slv_cancel_btn.setEnabled(False)
        self._tr_slv_progress.setVisible(False)
        self._slv_worker = None
        self._tr_slv_status.setText(f"Error: {exc}")

    def _tr_load_results(self) -> None:
        """Load tracks + tracked_labels into the napari viewer."""
        paths = self._get_paths()
        if paths is None:
            self._tr_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = out
        cfg = self._tr_build_config()
        msgs: list[str] = []

        try:
            tracks_df, graph = get_tracks_layer(wd, cfg)
            self.viewer.add_tracks(tracks_df.values, graph=graph, name="ultrack tracks")
            msgs.append(
                f"{tracks_df.iloc[:, 0].nunique()} tracks"
                f" ({len(tracks_df)} points)"
            )
        except Exception as e:
            msgs.append(f"tracks error: {e}")

        try:
            labels = get_labels_layer(wd)
            self.viewer.add_labels(labels, name="tracked labels")
            msgs.append(f"labels {labels.shape}")
        except FileNotFoundError:
            msgs.append("tracked_labels.tif not found")
        except Exception as e:
            msgs.append(f"labels error: {e}")

        self._tr_status.setText("Loaded: " + " | ".join(msgs))

    def _tr_on_load_results(self) -> None:
        self._tr_load_results()

    def _tr_on_export_ctc(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._tr_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = out
        output_dir = QFileDialog.getExistingDirectory(self, "Select CTC output directory")
        if not output_dir:
            return
        cfg = self._tr_build_config()
        try:
            export_ctc(wd, output_dir, cfg)
            self._tr_status.setText(f"CTC export written to {output_dir}")
        except Exception as e:
            self._tr_status.setText(f"CTC export error: {e}")

    # ── Database inspection methods ───────────────────────────────────────

    def _db_on_load_candidates(self) -> None:
        """Load all segmentation candidate nodes from the database."""
        paths = self._get_paths()
        if paths is None:
            self._db_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = Path(out)

        self._db_load_candidates_btn.setEnabled(False)
        self._db_status.setText("Loading candidates…")

        @thread_worker(connect={
            "returned": self._db_candidates_on_result,
            "errored": self._db_candidates_on_error,
        })
        def _work():
            db_path = wd / "data.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            conn = sqlite3.connect(str(db_path))
            query = """
                SELECT t, z, y, x, area, node_prob, frontier, height, selected
                FROM nodes
                ORDER BY t, id
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        self._db_worker = _work()

    def _db_candidates_on_result(self, df: pd.DataFrame) -> None:
        """Handle result from load_candidates query."""
        self._db_load_candidates_btn.setEnabled(True)
        self._db_candidates_df = df

        if df.empty:
            self._db_status.setText("No candidates found.")
            return

        # Create or update Points layer, projected to z=0
        coords = df[["t", "z", "y", "x"]].values.copy()
        coords[:, 1] = 0  # project to z=0

        # Color by selected status: green for selected, red for rejected (RGBA)
        selected = df["selected"].values.astype(bool)
        face_color = np.where(
            selected[:, None],
            [0.2, 0.9, 0.2, 1.0],  # green
            [0.9, 0.2, 0.2, 1.0],  # red
        )

        # Compute per-point sizes
        sizes = self._db_compute_sizes()

        layer_name = "candidates"
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            layer.data = coords
            layer.face_color = face_color
            layer.size = sizes
        else:
            self.viewer.add_points(
                coords,
                face_color=face_color,
                name=layer_name,
                size=sizes,
                opacity=0.8,
            )

        self._db_size_by_combo.setEnabled(True)
        self._db_size_scale.setEnabled(True)
        self._db_status.setText(
            f"Loaded {len(df)} candidates (green=selected, red=rejected). "
            f"Z-projected. Selected: {selected.sum()}, Rejected: {(~selected).sum()}"
        )

    def _db_candidates_on_error(self, exc: Exception) -> None:
        """Handle error from load_candidates query."""
        self._db_load_candidates_btn.setEnabled(True)
        self._db_status.setText(f"Error loading candidates: {exc}")

    def _db_on_load_links(self) -> None:
        """Load all candidate links as vectors."""
        paths = self._get_paths()
        if paths is None:
            self._db_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = Path(out)

        self._db_load_links_btn.setEnabled(False)
        self._db_status.setText("Loading links…")

        @thread_worker(connect={
            "returned": self._db_links_on_result,
            "errored": self._db_links_on_error,
        })
        def _work():
            db_path = wd / "data.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            conn = sqlite3.connect(str(db_path))
            query = """
                SELECT n1.t, n1.z, n1.y, n1.x,
                       n2.z - n1.z as dz, n2.y - n1.y as dy, n2.x - n1.x as dx,
                       l.weight
                FROM links l
                JOIN nodes n1 ON l.source_id = n1.id
                JOIN nodes n2 ON l.target_id = n2.id
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        self._db_worker = _work()

    def _db_links_on_result(self, df: pd.DataFrame) -> None:
        """Handle result from load_links query."""
        self._db_load_links_btn.setEnabled(True)

        if df.empty:
            self._db_status.setText("No links found.")
            return

        # Create vectors in napari format: (N, 2, D) where D is spatial dims
        # First row: start position [z, y, x]
        # Second row: direction [dz, dy, dx]
        # Project to z=0 by zeroing z component and z-direction
        starts = df[["z", "y", "x"]].values.copy()  # (N, 3)
        starts[:, 0] = 0  # project start z to 0
        directions = df[["dz", "dy", "dx"]].values.copy()  # (N, 3)
        directions[:, 0] = 0  # zero out z-direction (project to 2D)
        vectors = np.stack([starts, directions], axis=1)  # (N, 2, 3)

        # Normalize weight to 0-1 for colormap
        weights = df["weight"].values
        if weights.size > 0:
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                w_norm = (weights - w_min) / (w_max - w_min)
            else:
                w_norm = np.ones_like(weights) * 0.5
        else:
            w_norm = np.array([])

        # Color by weight using plasma colormap (N, 4) RGBA
        cmap = get_cmap("plasma")
        edge_color = cmap(w_norm)

        layer_name = "candidate_links"
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            layer.data = vectors
            layer.edge_color = edge_color
        else:
            self.viewer.add_vectors(
                vectors,
                edge_color=edge_color,
                edge_width=1,
                name=layer_name,
                opacity=0.7,
            )

        self._db_status.setText(f"Loaded {len(df)} candidate links (z-projected, colored by weight)")

    def _db_links_on_error(self, exc: Exception) -> None:
        """Handle error from load_links query."""
        self._db_load_links_btn.setEnabled(True)
        self._db_status.setText(f"Error loading links: {exc}")

    def _db_on_load_divisions(self) -> None:
        """Load division events (parent nodes with ≥2 selected children)."""
        paths = self._get_paths()
        if paths is None:
            self._db_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = Path(out)

        self._db_load_divisions_btn.setEnabled(False)
        self._db_status.setText("Loading divisions…")

        @thread_worker(connect={
            "returned": self._db_divisions_on_result,
            "errored": self._db_divisions_on_error,
        })
        def _work():
            db_path = wd / "data.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            conn = sqlite3.connect(str(db_path))
            query = """
                SELECT DISTINCT n.t, n.z, n.y, n.x
                FROM nodes n
                WHERE n.selected = 1
                AND (
                    SELECT COUNT(DISTINCT l.target_id)
                    FROM links l
                    JOIN nodes n_child ON l.target_id = n_child.id
                    WHERE l.source_id = n.id AND n_child.selected = 1
                ) >= 2
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        self._db_worker = _work()

    def _db_divisions_on_result(self, df: pd.DataFrame) -> None:
        """Handle result from load_divisions query."""
        self._db_load_divisions_btn.setEnabled(True)

        if df.empty:
            self._db_status.setText("No division events found.")
            return

        coords = df[["t", "z", "y", "x"]].values.copy()
        coords[:, 1] = 0  # project to z=0
        layer_name = "divisions"
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            layer.data = coords
        else:
            self.viewer.add_points(
                coords,
                face_color="magenta",
                name=layer_name,
                size=7,
                opacity=0.9,
            )

        self._db_status.setText(f"Loaded {len(df)} division events (z-projected)")

    def _db_divisions_on_error(self, exc: Exception) -> None:
        """Handle error from load_divisions query."""
        self._db_load_divisions_btn.setEnabled(True)
        self._db_status.setText(f"Error loading divisions: {exc}")

    def _db_compute_sizes(self) -> np.ndarray:
        """Compute per-point sizes based on current size_by column and scale."""
        if self._db_candidates_df is None:
            return np.array([])

        scale = self._db_size_scale.value()
        column = self._db_size_by_combo.currentText()
        n = len(self._db_candidates_df)

        if column == "none" or column not in self._db_candidates_df.columns:
            return np.full(n, scale)

        values = self._db_candidates_df[column].values.astype(float)
        v_min, v_max = np.nanmin(values), np.nanmax(values)
        if v_max > v_min:
            normalized = (values - v_min) / (v_max - v_min)
        else:
            normalized = np.ones(n) * 0.5

        # Scale from 20% to 100% of scale value
        return scale * (0.2 + 0.8 * normalized)

    def _db_on_size_by_changed(self, column: str) -> None:
        """Re-size candidates layer by a scalar column."""
        if self._db_candidates_df is None:
            return

        layer_name = "candidates"
        if layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]
        layer.size = self._db_compute_sizes()
        self._db_status.setText(f"Sized candidates by {column}")

    def _db_on_size_scale_changed(self, value: float) -> None:
        """Update overall point size scale."""
        if self._db_candidates_df is None:
            return

        layer_name = "candidates"
        if layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]
        layer.size = self._db_compute_sizes()

    def _db_on_load_labels(self) -> None:
        """Load tracked segmentation labels (tracked_labels.tif) as a Labels layer."""
        paths = self._get_paths()
        if paths is None:
            self._db_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = Path(out)
        labels_path = wd / "tracked_labels.tif"

        if not labels_path.exists():
            self._db_status.setText(f"Labels file not found: {labels_path}")
            return

        try:
            labels = tifffile.imread(str(labels_path))
            layer_name = "tracked_labels"
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                layer.data = labels
            else:
                self.viewer.add_labels(labels, name=layer_name)
            self._db_status.setText(f"Loaded tracked_labels {labels.shape}")
        except Exception as e:
            self._db_status.setText(f"Error loading labels: {e}")

    def _db_on_show_stats(self) -> None:
        """Query database for summary statistics."""
        paths = self._get_paths()
        if paths is None:
            self._db_status.setText("Set input and output directories first.")
            return
        _, out = paths
        wd = Path(out)

        self._db_show_stats_btn.setEnabled(False)
        self._db_status.setText("Loading statistics…")

        @thread_worker(connect={
            "returned": self._db_stats_on_result,
            "errored": self._db_stats_on_error,
        })
        def _work():
            db_path = wd / "data.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            conn = sqlite3.connect(str(db_path))

            # Query node statistics
            node_query = """
                SELECT
                    COUNT(*) AS total_nodes,
                    SUM(CASE WHEN selected=1 THEN 1 ELSE 0 END) AS selected_nodes,
                    COUNT(DISTINCT t) AS num_timepoints,
                    ROUND(MIN(area), 2) AS min_area,
                    ROUND(MAX(area), 2) AS max_area,
                    ROUND(AVG(area), 2) AS avg_area
                FROM nodes
            """
            node_stats = pd.read_sql_query(node_query, conn).iloc[0]

            # Query link statistics
            link_query = "SELECT COUNT(*) AS total_links FROM links"
            link_stats = pd.read_sql_query(link_query, conn).iloc[0]

            conn.close()
            return {"nodes": node_stats, "links": link_stats}

        self._db_worker = _work()

    def _db_stats_on_result(self, stats: dict) -> None:
        """Handle result from show_stats query."""
        self._db_show_stats_btn.setEnabled(True)
        node_stats = stats["nodes"]
        link_stats = stats["links"]

        text = (
            f"Nodes: {int(node_stats['total_nodes'])} total, "
            f"{int(node_stats['selected_nodes'])} selected\n"
            f"Timepoints: {int(node_stats['num_timepoints'])}\n"
            f"Area: min={node_stats['min_area']}, max={node_stats['max_area']}, "
            f"avg={node_stats['avg_area']}\n"
            f"Links: {int(link_stats['total_links'])} total"
        )
        self._db_status.setText(text)

    def _db_stats_on_error(self, exc: Exception) -> None:
        """Handle error from show_stats query."""
        self._db_show_stats_btn.setEnabled(True)
        self._db_status.setText(f"Error loading statistics: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Run All
    # ══════════════════════════════════════════════════════════════════════

    def _on_run_all(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._all_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        out_p = Path(out)
        fg_path = str(out_p / "foreground.tif")
        ct_path = str(out_p / "contours.tif")

        cp_ct_cfg = self._cp_ct_build_config()
        cp_ct_ow = self._cp_ct_overwrite_chk.isChecked()
        tr_cfg = self._tr_build_config()

        # Tracking skip: skip when all overwrite flags are False and outputs already exist
        tr_skip = (
            not tr_cfg.overwrite_segmentation
            and not tr_cfg.overwrite_linking
            and not tr_cfg.overwrite_solve
            and (out_p / "tracks.csv").exists()
            and (out_p / "tracked_labels.tif").exists()
        )

        self._run_all_btn.setEnabled(False)
        self._cancel_all_btn.setEnabled(True)
        self._all_progress.setVisible(True)
        self._all_status.setText("Starting Run All\u2026")

        out_path = Path(out)

        @thread_worker(connect={
            "yielded": self._on_all_progress,
            "finished": self._on_all_finished,
            "errored": self._on_all_error,
        })
        def _work():
            # Cellpose Contours
            if not cp_ct_ow and (out_path / "foreground.tif").exists() and (out_path / "contours.tif").exists():
                yield (0, 100, "[Cellpose Contours] Skipping \u2014 output exists (overwrite unchecked)")
            else:
                yield (0, 100, "[Cellpose Contours] Starting\u2026")
                for done, total, label in run_s02c(inp, out, cp_ct_cfg, overwrite=cp_ct_ow):
                    yield (int(done / max(total, 1) * 50), 100,
                           f"[Cellpose Contours] {label} [{done}/{total}]")

            # Tracking
            if tr_skip:
                yield (50, 100, "[Tracking] Skipping \u2014 output exists (overwrite=none)")
                yield (100, 100, "Run All complete.")
            else:
                yield (50, 100, "[Tracking] Starting\u2026")
                for step, total_steps, label in run_s03(fg_path, ct_path, out, tr_cfg):
                    yield (50 + int(step / max(total_steps, 1) * 50), 100,
                           f"[Tracking] {label}")
                yield (100, 100, "Run All complete.")

        self._all_worker = _work()
        self._all_worker.aborted.connect(self._on_all_cancelled)

    def _on_cancel_all(self) -> None:
        if self._all_worker is not None:
            self._all_worker.quit()

    def _on_all_cancelled(self) -> None:
        self._run_all_btn.setEnabled(True)
        self._cancel_all_btn.setEnabled(False)
        self._all_progress.setVisible(False)
        self._all_worker = None
        self._all_status.setText("Run All cancelled.")

    def _on_all_progress(self, u: tuple) -> None:
        done, total, label = u
        self._all_progress.setMaximum(max(total, 1))
        self._all_progress.setValue(done)
        self._all_status.setText(label)

    def _on_all_finished(self) -> None:
        self._run_all_btn.setEnabled(True)
        self._cancel_all_btn.setEnabled(False)
        self._all_progress.setVisible(False)
        self._all_worker = None
        self._all_status.setText("Run All complete \u2014 loading results\u2026")
        self._tr_load_results()

    def _on_all_error(self, exc: Exception) -> None:
        self._run_all_btn.setEnabled(True)
        self._cancel_all_btn.setEnabled(False)
        self._all_progress.setVisible(False)
        self._all_worker = None
        self._all_status.setText(f"Run All error: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Save / Load all parameters
    # ══════════════════════════════════════════════════════════════════════

    def _on_save_all_params(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save all parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = {
            "input_path": self._input_edit.text(),
            "output_path": self._output_edit.text(),
            "cp_contours": self._cp_ct_build_config().model_dump(),
            "tracking": self._tr_build_config().model_dump(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def _on_load_all_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load all parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        if "input_path" in data:
            self._input_edit.setText(data["input_path"])
        if "output_path" in data:
            self._output_edit.setText(data["output_path"])
        if "cp_contours" in data:
            self._cp_ct_apply_config(CellposeContoursConfig(**data["cp_contours"]))
        if "tracking" in data:
            self._tr_apply_config(TrackingConfig(**data["tracking"]))
