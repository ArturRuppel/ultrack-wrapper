"""Contour map panel — s02b interactive edge/boundary map for Ultrack."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

from ultrack_wrapper._config import ContoursConfig
from ultrack_wrapper.runners.terminal import launch_in_terminal
from ultrack_wrapper.stages.s02b_contours import (
    compute_contours_from_array,
    discover_prob_files,
    load_prob_map,
    run as run_s02b,
)


class ContoursWidget(QWidget):
    """Interactive contour/edge map generation for Ultrack."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None
        self._prob: np.ndarray | None = None
        self._contours_layer = None
        self._fg_layer = None

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._update_preview)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Cellpose output directory ────────────────────────────────────
        layout.addWidget(QLabel("Cellpose output directory (prob maps)"))
        row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("/path/to/1a_cellpose_nucleus")
        row.addWidget(self._input_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_input)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Output directory ─────────────────────────────────────────────
        layout.addWidget(QLabel("Output directory"))
        row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("/path/to/2b_contours")
        row.addWidget(self._output_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_output)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Timepoint selector ───────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview timepoint"))
        self._tp_spin = QSpinBox()
        self._tp_spin.setRange(0, 9999)
        self._tp_spin.setValue(0)
        row.addWidget(self._tp_spin)
        layout.addLayout(row)

        # ── Method ───────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["probmap", "watershed", "combined"])
        self._method_combo.setCurrentText("combined")
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        row.addWidget(self._method_combo)
        layout.addLayout(row)

        # ── Smooth sigma ─────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Smooth sigma"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 20.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._sigma_spin)
        layout.addLayout(row)

        # ── Combined weights ─────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("w_prob"))
        self._w_prob_spin = QDoubleSpinBox()
        self._w_prob_spin.setRange(0.0, 1.0)
        self._w_prob_spin.setSingleStep(0.1)
        self._w_prob_spin.setDecimals(2)
        self._w_prob_spin.setValue(0.4)
        self._w_prob_spin.valueChanged.connect(self._on_w_prob_changed)
        row.addWidget(self._w_prob_spin)
        row.addWidget(QLabel("w_ws"))
        self._w_ws_spin = QDoubleSpinBox()
        self._w_ws_spin.setRange(0.0, 1.0)
        self._w_ws_spin.setSingleStep(0.1)
        self._w_ws_spin.setDecimals(2)
        self._w_ws_spin.setValue(0.6)
        self._w_ws_spin.valueChanged.connect(self._on_w_ws_changed)
        row.addWidget(self._w_ws_spin)
        layout.addLayout(row)

        # ── Watershed parameters ─────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Min seed dist"))
        self._seed_dist_spin = QSpinBox()
        self._seed_dist_spin.setRange(1, 50)
        self._seed_dist_spin.setValue(5)
        self._seed_dist_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._seed_dist_spin)
        row.addWidget(QLabel("FG thresh"))
        self._fg_thresh_spin = QDoubleSpinBox()
        self._fg_thresh_spin.setRange(0.0, 1.0)
        self._fg_thresh_spin.setSingleStep(0.05)
        self._fg_thresh_spin.setDecimals(2)
        self._fg_thresh_spin.setValue(0.3)
        self._fg_thresh_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._fg_thresh_spin)
        layout.addLayout(row)

        # ── Preview button ───────────────────────────────────────────────
        row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        row.addWidget(self._preview_btn)
        layout.addLayout(row)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_btn = QPushButton("Run Contours")
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._run_btn)
        self._run_terminal_btn = QPushButton("Run in Terminal")
        self._run_terminal_btn.clicked.connect(self._on_run_terminal)
        row.addWidget(self._run_terminal_btn)
        layout.addLayout(row)

        # ── Overwrite + Load results ─────────────────────────────────────
        self._overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self._overwrite_check)

        self._load_results_btn = QPushButton("Load Results")
        self._load_results_btn.clicked.connect(self._on_load_results)
        layout.addWidget(self._load_results_btn)

        # ── Save / Load parameters ───────────────────────────────────────
        row = QHBoxLayout()
        save_btn = QPushButton("Save Parameters\u2026")
        save_btn.clicked.connect(self._on_save_params)
        row.addWidget(save_btn)
        load_btn = QPushButton("Load Parameters\u2026")
        load_btn.clicked.connect(self._on_load_params)
        row.addWidget(load_btn)
        layout.addLayout(row)

        # ── Progress ─────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._on_method_changed(self._method_combo.currentText())
        self.setLayout(layout)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select cellpose output directory")
        if d:
            self._input_edit.setText(d)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select contours output directory")
        if d:
            self._output_edit.setText(d)

    def _on_method_changed(self, method: str) -> None:
        is_combined = method == "combined"
        uses_ws = method in ("watershed", "combined")
        self._w_prob_spin.setEnabled(is_combined)
        self._w_ws_spin.setEnabled(is_combined)
        self._seed_dist_spin.setEnabled(uses_ws)
        self._fg_thresh_spin.setEnabled(uses_ws)
        self._schedule_update()

    def _on_w_prob_changed(self, val: float) -> None:
        self._w_ws_spin.blockSignals(True)
        self._w_ws_spin.setValue(round(1.0 - val, 2))
        self._w_ws_spin.blockSignals(False)
        self._schedule_update()

    def _on_w_ws_changed(self, val: float) -> None:
        self._w_prob_spin.blockSignals(True)
        self._w_prob_spin.setValue(round(1.0 - val, 2))
        self._w_prob_spin.blockSignals(False)
        self._schedule_update()

    def _build_config(self) -> ContoursConfig:
        return ContoursConfig(
            method=self._method_combo.currentText(),
            smooth_sigma=self._sigma_spin.value(),
            w_prob=self._w_prob_spin.value(),
            w_ws=self._w_ws_spin.value(),
            min_seed_dist=self._seed_dist_spin.value(),
            fg_thresh=self._fg_thresh_spin.value(),
        )

    def _schedule_update(self) -> None:
        if self._prob is not None:
            self._update_timer.start()

    # ── Preview ──────────────────────────────────────────────────────────

    def _on_preview(self) -> None:
        input_dir = self._input_edit.text().strip()
        if not input_dir:
            self._status_label.setText("Set cellpose output directory first.")
            return
        prob_files = discover_prob_files(input_dir)
        idx = self._tp_spin.value()
        if not prob_files:
            self._status_label.setText("No t*_prob.tif files found.")
            return
        if idx >= len(prob_files):
            self._status_label.setText(f"Only {len(prob_files)} timepoints available.")
            return
        self._status_label.setText(f"Loading {prob_files[idx].name}\u2026")
        self._prob = load_prob_map(prob_files[idx])
        self._update_preview()
        self._status_label.setText(f"Preview: {prob_files[idx].name}")

    def _update_preview(self) -> None:
        if self._prob is None:
            return
        cfg = self._build_config()
        contours, fg = compute_contours_from_array(self._prob, cfg)
        if self._contours_layer is None or self._contours_layer not in self.viewer.layers:
            self._contours_layer = self.viewer.add_image(
                contours, name="contours preview", colormap="hot",
            )
        else:
            self._contours_layer.data = contours
        if self._fg_layer is None or self._fg_layer not in self.viewer.layers:
            self._fg_layer = self.viewer.add_image(
                fg, name="sigmoid foreground", colormap="green", visible=False,
            )
        else:
            self._fg_layer.data = fg

    # ── Run contours ─────────────────────────────────────────────────────

    def _on_run(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        if not input_dir or not output_dir:
            self._status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_config()
        overwrite = self._overwrite_check.isChecked()
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Starting\u2026")

        @thread_worker(
            connect={
                "yielded": self._on_progress,
                "finished": self._on_finished,
                "errored": self._on_error,
            }
        )
        def _work():
            for update in run_s02b(input_dir, output_dir, cfg, overwrite=overwrite):
                yield update

        self._worker = _work()

    def _on_run_terminal(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        if not input_dir or not output_dir:
            self._status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_contours_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._overwrite_check.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s02b_contours"
            f" --input-dir \"{input_dir}\""
            f" --output-dir \"{output_dir}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._status_label.setText("Launched contours stage in terminal.")
        except Exception as e:
            self._status_label.setText(f"Terminal launch error: {e}")

    def _on_progress(self, update: tuple) -> None:
        done, total, label = update
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._status_label.setText(f"Processing {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("Done \u2014 contours.tif written.")
        self._worker = None
        self._load_contours_stack()

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None

    # ── Load results ────────────────────────────────────────────────────

    def _load_contours_stack(self) -> None:
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            return
        path = Path(output_dir) / "contours.tif"
        if not path.exists():
            self._status_label.setText("No contours.tif found in output directory.")
            return
        stack = tifffile.imread(str(path))
        self.viewer.add_image(stack, name="contours (all timepoints)", colormap="hot")
        self._status_label.setText(f"Loaded contours stack: {stack.shape}")

    def _on_load_results(self) -> None:
        self._load_contours_stack()

    # ── Save / Load parameters ──────────────────────────────────────────

    def _on_save_params(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        cfg = self._build_config()
        Path(path).write_text(json.dumps(cfg.model_dump(), indent=2))
        self._status_label.setText(f"Parameters saved to {Path(path).name}")

    def _on_load_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        cfg = ContoursConfig(**data)
        self._apply_config(cfg)
        self._status_label.setText(f"Parameters loaded from {Path(path).name}")

    def _apply_config(self, cfg: ContoursConfig) -> None:
        self._method_combo.setCurrentText(cfg.method)
        self._sigma_spin.setValue(cfg.smooth_sigma)
        self._w_prob_spin.setValue(cfg.w_prob)
        self._w_ws_spin.setValue(cfg.w_ws)
        self._seed_dist_spin.setValue(cfg.min_seed_dist)
        self._fg_thresh_spin.setValue(cfg.fg_thresh)
