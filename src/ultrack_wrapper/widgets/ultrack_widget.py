"""Unified scrollable analysis widget: Foreground → Contours → Tracking."""

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

from ultrack_wrapper._config import ContoursConfig, ForegroundConfig, TrackingConfig
from ultrack_wrapper.runners.terminal import launch_in_terminal
from ultrack_wrapper.stages.s02_foreground import (
    apply_blur,
    apply_clahe,
    compute_foreground_from_mag,
    discover_prob_files,
    load_prob_map,
    run as run_s02,
)
from ultrack_wrapper.stages.s02b_contours import (
    compute_contours_from_array,
    run as run_s02b,
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
    """Single scrollable panel for Foreground, Contours, and Tracking stages.

    All paths are derived from two shared inputs at the top:
    - **Input directory** — contains ``t*_prob.tif`` Cellpose output files.
    - **Output directory** — base folder; stage sub-dirs are created automatically:
      ``2_foreground/``, ``2b_contours/``, ``3_tracking/``.
    """

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

        # Background workers
        self._fg_worker = None
        self._ct_worker = None
        self._tr_worker = None
        self._seg_worker = None
        self._lnk_worker = None
        self._slv_worker = None
        self._all_worker = None

        # Cached preview data
        self._fg_mag: np.ndarray | None = None
        self._ct_prob: np.ndarray | None = None

        # Napari layer handles (preview)
        self._fg_mag_layer = None
        self._fg_preproc_layer = None
        self._fg_preview_layer = None
        self._ct_contours_layer = None
        self._ct_fg_layer = None

        # Debounce timers
        self._fg_timer = QTimer()
        self._fg_timer.setSingleShot(True)
        self._fg_timer.setInterval(200)
        self._fg_timer.timeout.connect(self._fg_update_preview)

        self._ct_timer = QTimer()
        self._ct_timer.setSingleShot(True)
        self._ct_timer.setInterval(200)
        self._ct_timer.timeout.connect(self._ct_update_preview)

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
        lay.addWidget(self._build_foreground_section())
        lay.addWidget(self._build_contours_section())
        lay.addWidget(self._build_tracking_section())

        # ── Run All ──────────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_all_btn = QPushButton(
            "Run All  (Foreground \u2192 Contours \u2192 Tracking)"
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
    # FOREGROUND section
    # ══════════════════════════════════════════════════════════════════════

    def _build_foreground_section(self) -> QGroupBox:
        grp = QGroupBox("Foreground")
        lay = QVBoxLayout()

        # Preview timepoint
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview timepoint"))
        self._fg_tp_spin = QSpinBox()
        self._fg_tp_spin.setRange(0, 9999)
        row.addWidget(self._fg_tp_spin)
        lay.addLayout(row)

        # Preprocessing
        pre = QGroupBox("Preprocessing")
        pre_lay = QVBoxLayout()

        row = QHBoxLayout()
        self._fg_median_chk = QCheckBox("Median filter, radius")
        self._fg_median_chk.toggled.connect(self._fg_schedule)
        row.addWidget(self._fg_median_chk)
        self._fg_median_spin = QSpinBox()
        self._fg_median_spin.setRange(1, 10)
        self._fg_median_spin.setValue(2)
        self._fg_median_spin.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_median_spin)
        pre_lay.addLayout(row)

        row = QHBoxLayout()
        self._fg_gauss_chk = QCheckBox("Gaussian filter, sigma")
        self._fg_gauss_chk.toggled.connect(self._fg_schedule)
        row.addWidget(self._fg_gauss_chk)
        self._fg_gauss_spin = QDoubleSpinBox()
        self._fg_gauss_spin.setRange(0.1, 20.0)
        self._fg_gauss_spin.setSingleStep(0.5)
        self._fg_gauss_spin.setDecimals(1)
        self._fg_gauss_spin.setValue(1.0)
        self._fg_gauss_spin.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_gauss_spin)
        pre_lay.addLayout(row)

        self._fg_clahe_chk = QCheckBox("CLAHE")
        self._fg_clahe_chk.toggled.connect(self._fg_schedule)
        pre_lay.addWidget(self._fg_clahe_chk)

        row = QHBoxLayout()
        row.addWidget(QLabel("Clip limit"))
        self._fg_clahe_clip = QDoubleSpinBox()
        self._fg_clahe_clip.setRange(0.001, 1.0)
        self._fg_clahe_clip.setSingleStep(0.005)
        self._fg_clahe_clip.setDecimals(3)
        self._fg_clahe_clip.setValue(0.01)
        self._fg_clahe_clip.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_clahe_clip)
        row.addWidget(QLabel("Kernel size (0=auto)"))
        self._fg_clahe_kernel = QSpinBox()
        self._fg_clahe_kernel.setRange(0, 512)
        self._fg_clahe_kernel.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_clahe_kernel)
        pre_lay.addLayout(row)

        pre.setLayout(pre_lay)
        lay.addWidget(pre)

        # Threshold method
        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold method"))
        self._fg_method_combo = QComboBox()
        self._fg_method_combo.addItems(["fixed", "otsu", "triangle", "sigmoid"])
        self._fg_method_combo.currentTextChanged.connect(self._fg_on_method_changed)
        row.addWidget(self._fg_method_combo)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold"))
        self._fg_thresh_spin = QDoubleSpinBox()
        self._fg_thresh_spin.setRange(0.0, 50.0)
        self._fg_thresh_spin.setSingleStep(0.1)
        self._fg_thresh_spin.setDecimals(2)
        self._fg_thresh_spin.setValue(1.0)
        self._fg_thresh_spin.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_thresh_spin)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sigmoid center"))
        self._fg_sig_center = QDoubleSpinBox()
        self._fg_sig_center.setRange(0.0, 50.0)
        self._fg_sig_center.setSingleStep(0.1)
        self._fg_sig_center.setDecimals(2)
        self._fg_sig_center.setValue(1.0)
        self._fg_sig_center.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_sig_center)
        row.addWidget(QLabel("steepness"))
        self._fg_sig_steep = QDoubleSpinBox()
        self._fg_sig_steep.setRange(0.1, 50.0)
        self._fg_sig_steep.setSingleStep(0.5)
        self._fg_sig_steep.setDecimals(1)
        self._fg_sig_steep.setValue(3.0)
        self._fg_sig_steep.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_sig_steep)
        lay.addLayout(row)

        # Post-processing
        post = QGroupBox("Post-processing")
        pp = QVBoxLayout()

        row = QHBoxLayout()
        self._fg_fill_chk = QCheckBox("Fill holes, max size (0=all)")
        self._fg_fill_chk.setChecked(True)
        self._fg_fill_chk.toggled.connect(self._fg_schedule)
        row.addWidget(self._fg_fill_chk)
        self._fg_fill_spin = QSpinBox()
        self._fg_fill_spin.setRange(0, 10_000_000)
        self._fg_fill_spin.setSingleStep(100)
        self._fg_fill_spin.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_fill_spin)
        pp.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Morphology"))
        self._fg_morpho_combo = QComboBox()
        self._fg_morpho_combo.addItems(["none", "opening", "closing"])
        self._fg_morpho_combo.currentTextChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_morpho_combo)
        row.addWidget(QLabel("radius"))
        self._fg_morpho_r = QSpinBox()
        self._fg_morpho_r.setRange(1, 10)
        self._fg_morpho_r.setValue(2)
        self._fg_morpho_r.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_morpho_r)
        pp.addLayout(row)

        row = QHBoxLayout()
        self._fg_rm_small_chk = QCheckBox("Remove small objects, min size")
        self._fg_rm_small_chk.setChecked(True)
        self._fg_rm_small_chk.toggled.connect(self._fg_schedule)
        row.addWidget(self._fg_rm_small_chk)
        self._fg_rm_small_spin = QSpinBox()
        self._fg_rm_small_spin.setRange(0, 1_000_000)
        self._fg_rm_small_spin.setSingleStep(100)
        self._fg_rm_small_spin.setValue(500)
        self._fg_rm_small_spin.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_rm_small_spin)
        pp.addLayout(row)

        row = QHBoxLayout()
        self._fg_area_chk = QCheckBox("Area filter")
        self._fg_area_chk.toggled.connect(self._fg_schedule)
        row.addWidget(self._fg_area_chk)
        row.addWidget(QLabel("min"))
        self._fg_area_min = QSpinBox()
        self._fg_area_min.setRange(0, 10_000_000)
        self._fg_area_min.setSingleStep(100)
        self._fg_area_min.setValue(100)
        self._fg_area_min.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_area_min)
        row.addWidget(QLabel("max"))
        self._fg_area_max = QSpinBox()
        self._fg_area_max.setRange(0, 10_000_000)
        self._fg_area_max.setSingleStep(1000)
        self._fg_area_max.setValue(100_000)
        self._fg_area_max.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_area_max)
        pp.addLayout(row)

        row = QHBoxLayout()
        self._fg_dist_chk = QCheckBox("Distance filter, min radius")
        self._fg_dist_chk.toggled.connect(self._fg_schedule)
        row.addWidget(self._fg_dist_chk)
        self._fg_dist_r = QDoubleSpinBox()
        self._fg_dist_r.setRange(1.0, 50.0)
        self._fg_dist_r.setSingleStep(0.5)
        self._fg_dist_r.setDecimals(1)
        self._fg_dist_r.setValue(3.0)
        self._fg_dist_r.valueChanged.connect(self._fg_schedule)
        row.addWidget(self._fg_dist_r)
        pp.addLayout(row)

        post.setLayout(pp)
        lay.addWidget(post)

        # Buttons
        row = QHBoxLayout()
        self._fg_preview_btn = QPushButton("Preview")
        self._fg_preview_btn.clicked.connect(self._fg_on_preview)
        row.addWidget(self._fg_preview_btn)
        self._fg_edit_btn = QPushButton("Edit Mask")
        self._fg_edit_btn.setEnabled(False)
        self._fg_edit_btn.clicked.connect(self._fg_on_edit_mask)
        row.addWidget(self._fg_edit_btn)
        lay.addLayout(row)

        self._fg_overwrite_chk = QCheckBox("Overwrite existing files")
        lay.addWidget(self._fg_overwrite_chk)

        row = QHBoxLayout()
        self._fg_run_btn = QPushButton("Run Foreground")
        self._fg_run_btn.clicked.connect(self._fg_on_run)
        row.addWidget(self._fg_run_btn)
        self._fg_term_btn = QPushButton("Run in Terminal")
        self._fg_term_btn.clicked.connect(self._fg_on_run_terminal)
        row.addWidget(self._fg_term_btn)
        self._fg_cancel_btn = QPushButton("Cancel")
        self._fg_cancel_btn.setEnabled(False)
        self._fg_cancel_btn.clicked.connect(self._fg_on_cancel)
        row.addWidget(self._fg_cancel_btn)
        lay.addLayout(row)

        self._fg_load_btn = QPushButton("Load Results")
        self._fg_load_btn.clicked.connect(self._fg_on_load_results)
        lay.addWidget(self._fg_load_btn)

        self._fg_progress = QProgressBar()
        self._fg_progress.setVisible(False)
        lay.addWidget(self._fg_progress)
        self._fg_status = QLabel("")
        lay.addWidget(self._fg_status)

        grp.setLayout(lay)

        # Initial control state
        self._fg_sig_center.setEnabled(False)
        self._fg_sig_steep.setEnabled(False)
        return grp

    # ── Foreground helpers ────────────────────────────────────────────────

    def _fg_build_config(self) -> ForegroundConfig:
        return ForegroundConfig(
            median_filter=self._fg_median_chk.isChecked(),
            median_radius=self._fg_median_spin.value(),
            gaussian_filter=self._fg_gauss_chk.isChecked(),
            gaussian_sigma=self._fg_gauss_spin.value(),
            clahe=self._fg_clahe_chk.isChecked(),
            clahe_clip_limit=self._fg_clahe_clip.value(),
            clahe_kernel_size=self._fg_clahe_kernel.value(),
            method=self._fg_method_combo.currentText(),
            threshold=self._fg_thresh_spin.value(),
            sigmoid_center=self._fg_sig_center.value(),
            sigmoid_steepness=self._fg_sig_steep.value(),
            fill_holes=self._fg_fill_chk.isChecked(),
            fill_holes_max_size=self._fg_fill_spin.value(),
            morpho_op=self._fg_morpho_combo.currentText(),
            morpho_radius=self._fg_morpho_r.value(),
            remove_small=self._fg_rm_small_chk.isChecked(),
            remove_small_min_size=self._fg_rm_small_spin.value(),
            area_filter=self._fg_area_chk.isChecked(),
            area_filter_min=self._fg_area_min.value(),
            area_filter_max=self._fg_area_max.value(),
            distance_filter=self._fg_dist_chk.isChecked(),
            distance_filter_min_radius=self._fg_dist_r.value(),
        )

    def _fg_apply_config(self, cfg: ForegroundConfig) -> None:
        self._fg_median_chk.setChecked(cfg.median_filter)
        self._fg_median_spin.setValue(cfg.median_radius)
        self._fg_gauss_chk.setChecked(cfg.gaussian_filter)
        self._fg_gauss_spin.setValue(cfg.gaussian_sigma)
        self._fg_clahe_chk.setChecked(cfg.clahe)
        self._fg_clahe_clip.setValue(cfg.clahe_clip_limit)
        self._fg_clahe_kernel.setValue(cfg.clahe_kernel_size)
        self._fg_method_combo.setCurrentText(cfg.method)
        self._fg_thresh_spin.setValue(cfg.threshold)
        self._fg_sig_center.setValue(cfg.sigmoid_center)
        self._fg_sig_steep.setValue(cfg.sigmoid_steepness)
        self._fg_fill_chk.setChecked(cfg.fill_holes)
        self._fg_fill_spin.setValue(cfg.fill_holes_max_size)
        self._fg_morpho_combo.setCurrentText(cfg.morpho_op)
        self._fg_morpho_r.setValue(cfg.morpho_radius)
        self._fg_rm_small_chk.setChecked(cfg.remove_small)
        self._fg_rm_small_spin.setValue(cfg.remove_small_min_size)
        self._fg_area_chk.setChecked(cfg.area_filter)
        self._fg_area_min.setValue(cfg.area_filter_min)
        self._fg_area_max.setValue(cfg.area_filter_max)
        self._fg_dist_chk.setChecked(cfg.distance_filter)
        self._fg_dist_r.setValue(cfg.distance_filter_min_radius)

    def _fg_on_method_changed(self, method: str) -> None:
        self._fg_thresh_spin.setEnabled(method == "fixed")
        self._fg_sig_center.setEnabled(method == "sigmoid")
        self._fg_sig_steep.setEnabled(method == "sigmoid")
        self._fg_schedule()

    def _fg_schedule(self) -> None:
        if self._fg_mag is not None:
            self._fg_timer.start()

    def _fg_on_preview(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._fg_status.setText("Set input and output directories first.")
            return
        inp, _ = paths
        prob_files = discover_prob_files(inp)
        idx = self._fg_tp_spin.value()
        if not prob_files:
            self._fg_status.setText("No t*_prob.tif files found.")
            return
        if idx >= len(prob_files):
            self._fg_status.setText(f"Only {len(prob_files)} timepoints available.")
            return
        self._fg_status.setText(f"Loading {prob_files[idx].name}\u2026")
        self._fg_mag = load_prob_map(prob_files[idx])
        if self._fg_mag_layer is None or self._fg_mag_layer not in self.viewer.layers:
            self._fg_mag_layer = self.viewer.add_image(
                self._fg_mag, name="cell probability", colormap="inferno"
            )
        else:
            self._fg_mag_layer.data = self._fg_mag
        self._fg_update_preview()
        self._fg_edit_btn.setEnabled(True)
        self._fg_status.setText(f"Preview: {prob_files[idx].name}")

    def _fg_update_preview(self) -> None:
        if self._fg_mag is None:
            return
        cfg = self._fg_build_config()
        has_preproc = cfg.median_filter or cfg.gaussian_filter or cfg.clahe
        if has_preproc:
            pre = apply_blur(self._fg_mag, cfg)
            pre = apply_clahe(pre, cfg)
            if self._fg_preproc_layer is None or self._fg_preproc_layer not in self.viewer.layers:
                self._fg_preproc_layer = self.viewer.add_image(
                    pre, name="preprocessed magnitude", colormap="magma"
                )
            else:
                self._fg_preproc_layer.data = pre
        elif self._fg_preproc_layer is not None and self._fg_preproc_layer in self.viewer.layers:
            self.viewer.layers.remove(self._fg_preproc_layer)
            self._fg_preproc_layer = None
        mask = compute_foreground_from_mag(self._fg_mag, cfg)
        if self._fg_preview_layer is None or self._fg_preview_layer not in self.viewer.layers:
            self._fg_preview_layer = self.viewer.add_labels(
                mask.astype(np.int32), name="foreground preview"
            )
        else:
            self._fg_preview_layer.data = mask.astype(np.int32)

    def _fg_on_edit_mask(self) -> None:
        if self._fg_preview_layer is not None and self._fg_preview_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self._fg_preview_layer
            self._fg_preview_layer.mode = "paint"
            self._fg_status.setText("Editing mask — use paint (1) / erase (0) tools.")

    def _fg_on_run(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._fg_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        fg_out = out
        cfg = self._fg_build_config()
        overwrite = self._fg_overwrite_chk.isChecked()
        self._fg_run_btn.setEnabled(False)
        self._fg_cancel_btn.setEnabled(True)
        self._fg_progress.setVisible(True)
        self._fg_status.setText("Starting\u2026")

        @thread_worker(connect={
            "yielded": self._fg_on_progress,
            "finished": self._fg_on_finished,
            "errored": self._fg_on_error,
        })
        def _work():
            for u in run_s02(inp, fg_out, cfg, overwrite=overwrite):
                yield u

        self._fg_worker = _work()
        self._fg_worker.aborted.connect(self._fg_on_cancelled)

    def _fg_on_run_terminal(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._fg_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        fg_out = out
        cfg = self._fg_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_fg_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._fg_overwrite_chk.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s02_foreground"
            f" --input-dir \"{inp}\""
            f" --output-dir \"{fg_out}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._fg_status.setText("Launched foreground in terminal.")
        except Exception as e:
            self._fg_status.setText(f"Terminal error: {e}")

    def _fg_on_progress(self, u: tuple) -> None:
        done, total, label = u
        self._fg_progress.setMaximum(max(total, 1))
        self._fg_progress.setValue(done)
        self._fg_status.setText(f"[{done}/{total}] {label}")

    def _fg_on_cancel(self) -> None:
        if self._fg_worker is not None:
            self._fg_worker.quit()

    def _fg_on_cancelled(self) -> None:
        self._fg_run_btn.setEnabled(True)
        self._fg_cancel_btn.setEnabled(False)
        self._fg_progress.setVisible(False)
        self._fg_worker = None
        self._fg_status.setText("Cancelled.")

    def _fg_on_finished(self) -> None:
        self._fg_run_btn.setEnabled(True)
        self._fg_cancel_btn.setEnabled(False)
        self._fg_progress.setVisible(False)
        self._fg_worker = None
        self._fg_status.setText("Done \u2014 foreground.tif written.")
        self._fg_load_stack()

    def _fg_on_error(self, exc: Exception) -> None:
        self._fg_run_btn.setEnabled(True)
        self._fg_cancel_btn.setEnabled(False)
        self._fg_progress.setVisible(False)
        self._fg_worker = None
        self._fg_status.setText(f"Error: {exc}")

    def _fg_load_stack(self) -> None:
        paths = self._get_paths()
        if paths is None:
            return
        p = Path(paths[1]) / "foreground.tif"
        if not p.exists():
            self._fg_status.setText("foreground.tif not found.")
            return
        stack = tifffile.imread(str(p))
        self.viewer.add_labels(stack.astype(np.int32), name="foreground")
        self._fg_status.setText(f"Loaded foreground: {stack.shape}")

    def _fg_on_load_results(self) -> None:
        self._fg_load_stack()

    # ══════════════════════════════════════════════════════════════════════
    # CONTOURS section
    # ══════════════════════════════════════════════════════════════════════

    def _build_contours_section(self) -> QGroupBox:
        grp = QGroupBox("Contours")
        lay = QVBoxLayout()

        # Preview timepoint
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview timepoint"))
        self._ct_tp_spin = QSpinBox()
        self._ct_tp_spin.setRange(0, 9999)
        row.addWidget(self._ct_tp_spin)
        lay.addLayout(row)

        # Method
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self._ct_method = QComboBox()
        self._ct_method.addItems(["probmap", "watershed", "combined"])
        self._ct_method.setCurrentText("combined")
        self._ct_method.currentTextChanged.connect(self._ct_on_method_changed)
        row.addWidget(self._ct_method)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Smooth sigma"))
        self._ct_sigma = QDoubleSpinBox()
        self._ct_sigma.setRange(0.0, 20.0)
        self._ct_sigma.setSingleStep(0.5)
        self._ct_sigma.setDecimals(1)
        self._ct_sigma.setValue(1.0)
        self._ct_sigma.valueChanged.connect(self._ct_schedule)
        row.addWidget(self._ct_sigma)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("w_prob"))
        self._ct_w_prob = QDoubleSpinBox()
        self._ct_w_prob.setRange(0.0, 1.0)
        self._ct_w_prob.setSingleStep(0.1)
        self._ct_w_prob.setDecimals(2)
        self._ct_w_prob.setValue(0.4)
        self._ct_w_prob.valueChanged.connect(self._ct_on_w_prob)
        row.addWidget(self._ct_w_prob)
        row.addWidget(QLabel("w_ws"))
        self._ct_w_ws = QDoubleSpinBox()
        self._ct_w_ws.setRange(0.0, 1.0)
        self._ct_w_ws.setSingleStep(0.1)
        self._ct_w_ws.setDecimals(2)
        self._ct_w_ws.setValue(0.6)
        self._ct_w_ws.valueChanged.connect(self._ct_on_w_ws)
        row.addWidget(self._ct_w_ws)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min seed dist"))
        self._ct_seed_dist = QSpinBox()
        self._ct_seed_dist.setRange(1, 50)
        self._ct_seed_dist.setValue(5)
        self._ct_seed_dist.valueChanged.connect(self._ct_schedule)
        row.addWidget(self._ct_seed_dist)
        row.addWidget(QLabel("FG thresh"))
        self._ct_fg_thresh = QDoubleSpinBox()
        self._ct_fg_thresh.setRange(0.0, 1.0)
        self._ct_fg_thresh.setSingleStep(0.05)
        self._ct_fg_thresh.setDecimals(2)
        self._ct_fg_thresh.setValue(0.3)
        self._ct_fg_thresh.valueChanged.connect(self._ct_schedule)
        row.addWidget(self._ct_fg_thresh)
        lay.addLayout(row)

        # Buttons
        row = QHBoxLayout()
        self._ct_preview_btn = QPushButton("Preview")
        self._ct_preview_btn.clicked.connect(self._ct_on_preview)
        row.addWidget(self._ct_preview_btn)
        lay.addLayout(row)

        self._ct_overwrite_chk = QCheckBox("Overwrite existing files")
        lay.addWidget(self._ct_overwrite_chk)

        row = QHBoxLayout()
        self._ct_run_btn = QPushButton("Run Contours")
        self._ct_run_btn.clicked.connect(self._ct_on_run)
        row.addWidget(self._ct_run_btn)
        self._ct_term_btn = QPushButton("Run in Terminal")
        self._ct_term_btn.clicked.connect(self._ct_on_run_terminal)
        row.addWidget(self._ct_term_btn)
        self._ct_cancel_btn = QPushButton("Cancel")
        self._ct_cancel_btn.setEnabled(False)
        self._ct_cancel_btn.clicked.connect(self._ct_on_cancel)
        row.addWidget(self._ct_cancel_btn)
        lay.addLayout(row)

        self._ct_load_btn = QPushButton("Load Results")
        self._ct_load_btn.clicked.connect(self._ct_on_load_results)
        lay.addWidget(self._ct_load_btn)

        self._ct_progress = QProgressBar()
        self._ct_progress.setVisible(False)
        lay.addWidget(self._ct_progress)
        self._ct_status = QLabel("")
        lay.addWidget(self._ct_status)

        grp.setLayout(lay)
        self._ct_on_method_changed(self._ct_method.currentText())
        return grp

    # ── Contours helpers ──────────────────────────────────────────────────

    def _ct_build_config(self) -> ContoursConfig:
        return ContoursConfig(
            method=self._ct_method.currentText(),
            smooth_sigma=self._ct_sigma.value(),
            w_prob=self._ct_w_prob.value(),
            w_ws=self._ct_w_ws.value(),
            min_seed_dist=self._ct_seed_dist.value(),
            fg_thresh=self._ct_fg_thresh.value(),
        )

    def _ct_apply_config(self, cfg: ContoursConfig) -> None:
        self._ct_method.setCurrentText(cfg.method)
        self._ct_sigma.setValue(cfg.smooth_sigma)
        self._ct_w_prob.setValue(cfg.w_prob)
        self._ct_w_ws.setValue(cfg.w_ws)
        self._ct_seed_dist.setValue(cfg.min_seed_dist)
        self._ct_fg_thresh.setValue(cfg.fg_thresh)

    def _ct_on_method_changed(self, method: str) -> None:
        is_combined = method == "combined"
        uses_ws = method in ("watershed", "combined")
        self._ct_w_prob.setEnabled(is_combined)
        self._ct_w_ws.setEnabled(is_combined)
        self._ct_seed_dist.setEnabled(uses_ws)
        self._ct_fg_thresh.setEnabled(uses_ws)
        self._ct_schedule()

    def _ct_on_w_prob(self, val: float) -> None:
        self._ct_w_ws.blockSignals(True)
        self._ct_w_ws.setValue(round(1.0 - val, 2))
        self._ct_w_ws.blockSignals(False)
        self._ct_schedule()

    def _ct_on_w_ws(self, val: float) -> None:
        self._ct_w_prob.blockSignals(True)
        self._ct_w_prob.setValue(round(1.0 - val, 2))
        self._ct_w_prob.blockSignals(False)
        self._ct_schedule()

    def _ct_schedule(self) -> None:
        if self._ct_prob is not None:
            self._ct_timer.start()

    def _ct_on_preview(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._ct_status.setText("Set input and output directories first.")
            return
        inp, _ = paths
        from ultrack_wrapper.stages.s02b_contours import discover_prob_files as disc, load_prob_map as lpm
        prob_files = disc(inp)
        idx = self._ct_tp_spin.value()
        if not prob_files:
            self._ct_status.setText("No t*_prob.tif files found.")
            return
        if idx >= len(prob_files):
            self._ct_status.setText(f"Only {len(prob_files)} timepoints available.")
            return
        self._ct_status.setText(f"Loading {prob_files[idx].name}\u2026")
        self._ct_prob = lpm(prob_files[idx])
        self._ct_update_preview()
        self._ct_status.setText(f"Preview: {prob_files[idx].name}")

    def _ct_update_preview(self) -> None:
        if self._ct_prob is None:
            return
        cfg = self._ct_build_config()
        contours, fg = compute_contours_from_array(self._ct_prob, cfg)
        if self._ct_contours_layer is None or self._ct_contours_layer not in self.viewer.layers:
            self._ct_contours_layer = self.viewer.add_image(
                contours, name="contours preview", colormap="hot"
            )
        else:
            self._ct_contours_layer.data = contours
        if self._ct_fg_layer is None or self._ct_fg_layer not in self.viewer.layers:
            self._ct_fg_layer = self.viewer.add_image(
                fg, name="sigmoid foreground", colormap="green", visible=False
            )
        else:
            self._ct_fg_layer.data = fg

    def _ct_on_run(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._ct_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        ct_out = out
        cfg = self._ct_build_config()
        overwrite = self._ct_overwrite_chk.isChecked()
        self._ct_run_btn.setEnabled(False)
        self._ct_cancel_btn.setEnabled(True)
        self._ct_progress.setVisible(True)
        self._ct_status.setText("Starting\u2026")

        @thread_worker(connect={
            "yielded": self._ct_on_progress,
            "finished": self._ct_on_finished,
            "errored": self._ct_on_error,
        })
        def _work():
            for u in run_s02b(inp, ct_out, cfg, overwrite=overwrite):
                yield u

        self._ct_worker = _work()
        self._ct_worker.aborted.connect(self._ct_on_cancelled)

    def _ct_on_run_terminal(self) -> None:
        paths = self._get_paths()
        if paths is None:
            self._ct_status.setText("Set input and output directories first.")
            return
        inp, out = paths
        ct_out = out
        cfg = self._ct_build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_ct_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._ct_overwrite_chk.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s02b_contours"
            f" --input-dir \"{inp}\""
            f" --output-dir \"{ct_out}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._ct_status.setText("Launched contours in terminal.")
        except Exception as e:
            self._ct_status.setText(f"Terminal error: {e}")

    def _ct_on_progress(self, u: tuple) -> None:
        done, total, label = u
        self._ct_progress.setMaximum(max(total, 1))
        self._ct_progress.setValue(done)
        self._ct_status.setText(f"[{done}/{total}] {label}")

    def _ct_on_cancel(self) -> None:
        if self._ct_worker is not None:
            self._ct_worker.quit()

    def _ct_on_cancelled(self) -> None:
        self._ct_run_btn.setEnabled(True)
        self._ct_cancel_btn.setEnabled(False)
        self._ct_progress.setVisible(False)
        self._ct_worker = None
        self._ct_status.setText("Cancelled.")

    def _ct_on_finished(self) -> None:
        self._ct_run_btn.setEnabled(True)
        self._ct_cancel_btn.setEnabled(False)
        self._ct_progress.setVisible(False)
        self._ct_worker = None
        self._ct_status.setText("Done \u2014 contours.tif written.")
        self._ct_load_stack()

    def _ct_on_error(self, exc: Exception) -> None:
        self._ct_run_btn.setEnabled(True)
        self._ct_cancel_btn.setEnabled(False)
        self._ct_progress.setVisible(False)
        self._ct_worker = None
        self._ct_status.setText(f"Error: {exc}")

    def _ct_load_stack(self) -> None:
        paths = self._get_paths()
        if paths is None:
            return
        p = Path(paths[1]) / "contours.tif"
        if not p.exists():
            self._ct_status.setText("contours.tif not found.")
            return
        stack = tifffile.imread(str(p))
        self.viewer.add_image(stack, name="contours", colormap="hot")
        self._ct_status.setText(f"Loaded contours: {stack.shape}")

    def _ct_on_load_results(self) -> None:
        self._ct_load_stack()

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
        row.addWidget(self._tr_overwrite_seg_chk)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min area"))
        self._tr_min_area = QSpinBox()
        self._tr_min_area.setRange(1, 10_000_000)
        self._tr_min_area.setValue(100)
        row.addWidget(self._tr_min_area)
        row.addWidget(QLabel("Max area"))
        self._tr_max_area = QSpinBox()
        self._tr_max_area.setRange(1, 10_000_000)
        self._tr_max_area.setValue(1_000_000)
        row.addWidget(self._tr_max_area)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min frontier"))
        self._tr_min_front = QDoubleSpinBox()
        self._tr_min_front.setRange(0.0, 1.0)
        self._tr_min_front.setSingleStep(0.05)
        self._tr_min_front.setDecimals(3)
        row.addWidget(self._tr_min_front)
        row.addWidget(QLabel("FG threshold"))
        self._tr_seg_thresh = QDoubleSpinBox()
        self._tr_seg_thresh.setRange(0.0, 1.0)
        self._tr_seg_thresh.setSingleStep(0.05)
        self._tr_seg_thresh.setDecimals(2)
        self._tr_seg_thresh.setValue(0.5)
        row.addWidget(self._tr_seg_thresh)
        sl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("WS hierarchy"))
        self._tr_ws_combo = QComboBox()
        self._tr_ws_combo.addItems(["area", "dynamics", "volume"])
        row.addWidget(self._tr_ws_combo)
        row.addWidget(QLabel("Aniso. pen."))
        self._tr_aniso = QDoubleSpinBox()
        self._tr_aniso.setRange(-10.0, 10.0)
        self._tr_aniso.setSingleStep(0.1)
        self._tr_aniso.setDecimals(1)
        row.addWidget(self._tr_aniso)
        sl.addLayout(row)

        # Workers for segmentation
        row = QHBoxLayout()
        row.addWidget(QLabel("Workers"))
        self._tr_seg_workers = QSpinBox()
        self._tr_seg_workers.setRange(1, 64)
        self._tr_seg_workers.setValue(1)
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
        row.addWidget(self._tr_overwrite_lnk_chk)
        ll.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max distance"))
        self._tr_max_dist = QDoubleSpinBox()
        self._tr_max_dist.setRange(0.1, 500.0)
        self._tr_max_dist.setSingleStep(1.0)
        self._tr_max_dist.setDecimals(1)
        self._tr_max_dist.setValue(15.0)
        row.addWidget(self._tr_max_dist)
        row.addWidget(QLabel("Max neighbors"))
        self._tr_max_nb = QSpinBox()
        self._tr_max_nb.setRange(1, 50)
        self._tr_max_nb.setValue(5)
        row.addWidget(self._tr_max_nb)
        ll.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Distance weight"))
        self._tr_dist_w = QDoubleSpinBox()
        self._tr_dist_w.setRange(0.0, 10.0)
        self._tr_dist_w.setSingleStep(0.1)
        self._tr_dist_w.setDecimals(2)
        row.addWidget(self._tr_dist_w)
        ll.addLayout(row)

        # Workers for linking
        row = QHBoxLayout()
        row.addWidget(QLabel("Workers"))
        self._tr_lnk_workers = QSpinBox()
        self._tr_lnk_workers.setRange(1, 64)
        self._tr_lnk_workers.setValue(1)
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
        row.addWidget(self._tr_overwrite_slv_chk)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Appear"))
        self._tr_appear = QDoubleSpinBox()
        self._tr_appear.setRange(-100.0, 0.0)
        self._tr_appear.setSingleStep(0.001)
        self._tr_appear.setDecimals(4)
        self._tr_appear.setValue(-0.001)
        row.addWidget(self._tr_appear)
        row.addWidget(QLabel("Disappear"))
        self._tr_disappear = QDoubleSpinBox()
        self._tr_disappear.setRange(-100.0, 0.0)
        self._tr_disappear.setSingleStep(0.001)
        self._tr_disappear.setDecimals(4)
        self._tr_disappear.setValue(-0.001)
        row.addWidget(self._tr_disappear)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Division"))
        self._tr_division = QDoubleSpinBox()
        self._tr_division.setRange(-100.0, 0.0)
        self._tr_division.setSingleStep(0.001)
        self._tr_division.setDecimals(4)
        self._tr_division.setValue(-0.001)
        row.addWidget(self._tr_division)
        row.addWidget(QLabel("Link func"))
        self._tr_link_func = QComboBox()
        self._tr_link_func.addItems(["power", "identity"])
        row.addWidget(self._tr_link_func)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Power"))
        self._tr_power = QDoubleSpinBox()
        self._tr_power.setRange(0.1, 20.0)
        self._tr_power.setSingleStep(0.5)
        self._tr_power.setDecimals(1)
        self._tr_power.setValue(4.0)
        row.addWidget(self._tr_power)
        row.addWidget(QLabel("Bias"))
        self._tr_bias = QDoubleSpinBox()
        self._tr_bias.setRange(-10.0, 0.0)
        self._tr_bias.setSingleStep(0.01)
        self._tr_bias.setDecimals(3)
        row.addWidget(self._tr_bias)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gap"))
        self._tr_gap = QDoubleSpinBox()
        self._tr_gap.setRange(0.0, 1.0)
        self._tr_gap.setSingleStep(0.001)
        self._tr_gap.setDecimals(4)
        self._tr_gap.setValue(0.001)
        row.addWidget(self._tr_gap)
        row.addWidget(QLabel("Time limit (s)"))
        self._tr_time_limit = QSpinBox()
        self._tr_time_limit.setRange(10, 360_000)
        self._tr_time_limit.setValue(36_000)
        row.addWidget(self._tr_time_limit)
        sv.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Window size (0=all)"))
        self._tr_window = QSpinBox()
        self._tr_window.setRange(0, 10_000)
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
        row.addWidget(QLabel("Colour candidates by:"))
        self._db_colour_by_combo = QComboBox()
        self._db_colour_by_combo.addItems(["selected", "area", "node_prob", "frontier", "height"])
        self._db_colour_by_combo.setEnabled(False)
        self._db_colour_by_combo.currentTextChanged.connect(self._db_on_colour_by_changed)
        row.addWidget(self._db_colour_by_combo)
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

        # Create or update Points layer
        coords = df[["t", "z", "y", "x"]].values
        # Color by selected status: green for selected, red for rejected (RGBA)
        selected = df["selected"].values.astype(bool)
        face_color = np.where(
            selected[:, None],
            [0.2, 0.9, 0.2, 1.0],  # green
            [0.9, 0.2, 0.2, 1.0],  # red
        )

        layer_name = "candidates"
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            layer.data = coords
            layer.face_color = face_color
        else:
            self.viewer.add_points(
                coords,
                face_color=face_color,
                name=layer_name,
                size=5,
                opacity=0.8,
            )

        self._db_colour_by_combo.setEnabled(True)
        self._db_status.setText(f"Loaded {len(df)} candidates (green=selected, red=rejected)")

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
        starts = df[["z", "y", "x"]].values  # (N, 3)
        directions = df[["dz", "dy", "dx"]].values  # (N, 3)
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

        self._db_status.setText(f"Loaded {len(df)} candidate links (colored by weight)")

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

        coords = df[["t", "z", "y", "x"]].values
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

        self._db_status.setText(f"Loaded {len(df)} division events")

    def _db_divisions_on_error(self, exc: Exception) -> None:
        """Handle error from load_divisions query."""
        self._db_load_divisions_btn.setEnabled(True)
        self._db_status.setText(f"Error loading divisions: {exc}")

    def _db_on_colour_by_changed(self, column: str) -> None:
        """Re-color candidates layer by a different scalar column."""
        if self._db_candidates_df is None or column not in self._db_candidates_df.columns:
            return

        layer_name = "candidates"
        if layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]

        if column == "selected":
            # Binary color: green for True, red for False
            selected = self._db_candidates_df["selected"].values.astype(bool)
            face_color = np.where(
                selected[:, None],
                [0.2, 0.9, 0.2, 1.0],  # green
                [0.9, 0.2, 0.2, 1.0],  # red
            )
        else:
            # Continuous column: normalize and apply viridis colormap
            values = self._db_candidates_df[column].values.astype(float)
            v_min, v_max = np.nanmin(values), np.nanmax(values)
            if v_max > v_min:
                normalized = (values - v_min) / (v_max - v_min)
            else:
                normalized = np.ones_like(values) * 0.5
            cmap = get_cmap("viridis")
            face_color = cmap(normalized)

        layer.face_color = face_color
        self._db_status.setText(f"Colored candidates by {column}")

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

        fg_cfg = self._fg_build_config()
        fg_ow = self._fg_overwrite_chk.isChecked()
        ct_cfg = self._ct_build_config()
        ct_ow = self._ct_overwrite_chk.isChecked()
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
            # Foreground
            if not fg_ow and (out_path / "foreground.tif").exists():
                yield (0, 100, "[Foreground] Skipping \u2014 output exists (overwrite unchecked)")
            else:
                yield (0, 100, "[Foreground] Starting\u2026")
                for done, total, label in run_s02(inp, out, fg_cfg, overwrite=fg_ow):
                    yield (int(done / max(total, 1) * 30), 100,
                           f"[Foreground] {label} [{done}/{total}]")

            # Contours
            if not ct_ow and (out_path / "contours.tif").exists():
                yield (30, 100, "[Contours] Skipping \u2014 output exists (overwrite unchecked)")
            else:
                yield (30, 100, "[Contours] Starting\u2026")
                for done, total, label in run_s02b(inp, out, ct_cfg, overwrite=ct_ow):
                    yield (30 + int(done / max(total, 1) * 30), 100,
                           f"[Contours] {label} [{done}/{total}]")

            # Tracking
            if tr_skip:
                yield (60, 100, "[Tracking] Skipping \u2014 output exists (overwrite=none)")
                yield (100, 100, "Run All complete.")
            else:
                yield (60, 100, "[Tracking] Starting\u2026")
                for step, total_steps, label in run_s03(fg_path, ct_path, out, tr_cfg):
                    yield (60 + int(step / max(total_steps, 1) * 40), 100,
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
            "foreground": self._fg_build_config().model_dump(),
            "contours": self._ct_build_config().model_dump(),
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
        if "foreground" in data:
            self._fg_apply_config(ForegroundConfig(**data["foreground"]))
        if "contours" in data:
            self._ct_apply_config(ContoursConfig(**data["contours"]))
        if "tracking" in data:
            self._tr_apply_config(TrackingConfig(**data["tracking"]))
