"""Foreground detection panel — s02 interactive thresholding + post-processing."""

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

from ultrack_wrapper._config import ForegroundConfig
from ultrack_wrapper.runners.terminal import launch_in_terminal
from ultrack_wrapper.stages.s02_foreground import (
    apply_blur,
    apply_clahe,
    compute_foreground_from_mag,
    discover_prob_files,
    load_prob_map,
    run as run_s02,
)


class ForegroundWidget(QWidget):
    """Interactive foreground detection from Cellpose flow outputs."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None
        self._mag: np.ndarray | None = None
        self._preview_layer = None
        self._mag_layer = None
        self._clahe_layer = None

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._update_preview)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Input directory ──────────────────────────────────────────────
        layout.addWidget(QLabel("Cellpose output directory"))
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
        self._output_edit.setPlaceholderText("/path/to/2_foreground")
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

        # ── Preprocessing group ──────────────────────────────────────────
        preproc_group = QGroupBox("Preprocessing")
        pre_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._median_check = QCheckBox("Median filter, radius")
        self._median_check.toggled.connect(self._schedule_update)
        row.addWidget(self._median_check)
        self._median_radius_spin = QSpinBox()
        self._median_radius_spin.setRange(1, 10)
        self._median_radius_spin.setValue(2)
        self._median_radius_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._median_radius_spin)
        pre_layout.addLayout(row)

        row = QHBoxLayout()
        self._gaussian_check = QCheckBox("Gaussian filter, sigma")
        self._gaussian_check.toggled.connect(self._schedule_update)
        row.addWidget(self._gaussian_check)
        self._gaussian_sigma_spin = QDoubleSpinBox()
        self._gaussian_sigma_spin.setRange(0.1, 20.0)
        self._gaussian_sigma_spin.setSingleStep(0.5)
        self._gaussian_sigma_spin.setDecimals(1)
        self._gaussian_sigma_spin.setValue(1.0)
        self._gaussian_sigma_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._gaussian_sigma_spin)
        pre_layout.addLayout(row)

        self._clahe_check = QCheckBox("CLAHE")
        self._clahe_check.toggled.connect(self._schedule_update)
        pre_layout.addWidget(self._clahe_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Clip limit"))
        self._clahe_clip_spin = QDoubleSpinBox()
        self._clahe_clip_spin.setRange(0.001, 1.0)
        self._clahe_clip_spin.setSingleStep(0.005)
        self._clahe_clip_spin.setDecimals(3)
        self._clahe_clip_spin.setValue(0.01)
        self._clahe_clip_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._clahe_clip_spin)
        row.addWidget(QLabel("Kernel size (0=auto)"))
        self._clahe_kernel_spin = QSpinBox()
        self._clahe_kernel_spin.setRange(0, 512)
        self._clahe_kernel_spin.setValue(0)
        self._clahe_kernel_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._clahe_kernel_spin)
        pre_layout.addLayout(row)

        preproc_group.setLayout(pre_layout)
        layout.addWidget(preproc_group)

        # ── Threshold method ─────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold method"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["fixed", "otsu", "triangle", "sigmoid"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        row.addWidget(self._method_combo)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold"))
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.0, 50.0)
        self._thresh_spin.setSingleStep(0.1)
        self._thresh_spin.setDecimals(2)
        self._thresh_spin.setValue(1.0)
        self._thresh_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._thresh_spin)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sigmoid center"))
        self._sigmoid_center_spin = QDoubleSpinBox()
        self._sigmoid_center_spin.setRange(0.0, 50.0)
        self._sigmoid_center_spin.setSingleStep(0.1)
        self._sigmoid_center_spin.setDecimals(2)
        self._sigmoid_center_spin.setValue(1.0)
        self._sigmoid_center_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._sigmoid_center_spin)
        row.addWidget(QLabel("steepness"))
        self._sigmoid_steepness_spin = QDoubleSpinBox()
        self._sigmoid_steepness_spin.setRange(0.1, 50.0)
        self._sigmoid_steepness_spin.setSingleStep(0.5)
        self._sigmoid_steepness_spin.setDecimals(1)
        self._sigmoid_steepness_spin.setValue(3.0)
        self._sigmoid_steepness_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._sigmoid_steepness_spin)
        layout.addLayout(row)

        # ── Post-processing group ────────────────────────────────────────
        postproc_group = QGroupBox("Post-processing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._fill_holes_check = QCheckBox("Fill holes, max size (0=all)")
        self._fill_holes_check.setChecked(True)
        self._fill_holes_check.toggled.connect(self._schedule_update)
        row.addWidget(self._fill_holes_check)
        self._fill_holes_max_spin = QSpinBox()
        self._fill_holes_max_spin.setRange(0, 10_000_000)
        self._fill_holes_max_spin.setSingleStep(100)
        self._fill_holes_max_spin.setValue(0)
        self._fill_holes_max_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._fill_holes_max_spin)
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Morphology"))
        self._morpho_combo = QComboBox()
        self._morpho_combo.addItems(["none", "opening", "closing"])
        self._morpho_combo.currentTextChanged.connect(self._schedule_update)
        row.addWidget(self._morpho_combo)
        row.addWidget(QLabel("radius"))
        self._morpho_radius_spin = QSpinBox()
        self._morpho_radius_spin.setRange(1, 10)
        self._morpho_radius_spin.setValue(2)
        self._morpho_radius_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._morpho_radius_spin)
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        self._remove_small_check = QCheckBox("Remove small objects, min size")
        self._remove_small_check.setChecked(True)
        self._remove_small_check.toggled.connect(self._schedule_update)
        row.addWidget(self._remove_small_check)
        self._remove_small_spin = QSpinBox()
        self._remove_small_spin.setRange(0, 1_000_000)
        self._remove_small_spin.setSingleStep(100)
        self._remove_small_spin.setValue(500)
        self._remove_small_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._remove_small_spin)
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        self._area_filter_check = QCheckBox("Area filter")
        self._area_filter_check.toggled.connect(self._schedule_update)
        row.addWidget(self._area_filter_check)
        row.addWidget(QLabel("min"))
        self._area_min_spin = QSpinBox()
        self._area_min_spin.setRange(0, 10_000_000)
        self._area_min_spin.setSingleStep(100)
        self._area_min_spin.setValue(100)
        self._area_min_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._area_min_spin)
        row.addWidget(QLabel("max"))
        self._area_max_spin = QSpinBox()
        self._area_max_spin.setRange(0, 10_000_000)
        self._area_max_spin.setSingleStep(1000)
        self._area_max_spin.setValue(100_000)
        self._area_max_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._area_max_spin)
        pp_layout.addLayout(row)

        row = QHBoxLayout()
        self._distance_filter_check = QCheckBox("Distance filter, min radius")
        self._distance_filter_check.toggled.connect(self._schedule_update)
        row.addWidget(self._distance_filter_check)
        self._distance_radius_spin = QDoubleSpinBox()
        self._distance_radius_spin.setRange(1.0, 50.0)
        self._distance_radius_spin.setSingleStep(0.5)
        self._distance_radius_spin.setDecimals(1)
        self._distance_radius_spin.setValue(3.0)
        self._distance_radius_spin.valueChanged.connect(self._schedule_update)
        row.addWidget(self._distance_radius_spin)
        pp_layout.addLayout(row)

        postproc_group.setLayout(pp_layout)
        layout.addWidget(postproc_group)

        # ── Preview buttons ──────────────────────────────────────────────
        row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        row.addWidget(self._preview_btn)
        self._edit_btn = QPushButton("Edit Mask")
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit_mask)
        row.addWidget(self._edit_btn)
        layout.addLayout(row)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_btn = QPushButton("Run Foreground")
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

        self._sigmoid_center_spin.setEnabled(False)
        self._sigmoid_steepness_spin.setEnabled(False)

        self.setLayout(layout)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select cellpose output directory")
        if d:
            self._input_edit.setText(d)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select foreground output directory")
        if d:
            self._output_edit.setText(d)

    def _on_method_changed(self, method: str) -> None:
        self._thresh_spin.setEnabled(method == "fixed")
        self._sigmoid_center_spin.setEnabled(method == "sigmoid")
        self._sigmoid_steepness_spin.setEnabled(method == "sigmoid")
        self._schedule_update()

    def _build_config(self) -> ForegroundConfig:
        return ForegroundConfig(
            median_filter=self._median_check.isChecked(),
            median_radius=self._median_radius_spin.value(),
            gaussian_filter=self._gaussian_check.isChecked(),
            gaussian_sigma=self._gaussian_sigma_spin.value(),
            clahe=self._clahe_check.isChecked(),
            clahe_clip_limit=self._clahe_clip_spin.value(),
            clahe_kernel_size=self._clahe_kernel_spin.value(),
            method=self._method_combo.currentText(),
            threshold=self._thresh_spin.value(),
            sigmoid_center=self._sigmoid_center_spin.value(),
            sigmoid_steepness=self._sigmoid_steepness_spin.value(),
            fill_holes=self._fill_holes_check.isChecked(),
            fill_holes_max_size=self._fill_holes_max_spin.value(),
            morpho_op=self._morpho_combo.currentText(),
            morpho_radius=self._morpho_radius_spin.value(),
            remove_small=self._remove_small_check.isChecked(),
            remove_small_min_size=self._remove_small_spin.value(),
            area_filter=self._area_filter_check.isChecked(),
            area_filter_min=self._area_min_spin.value(),
            area_filter_max=self._area_max_spin.value(),
            distance_filter=self._distance_filter_check.isChecked(),
            distance_filter_min_radius=self._distance_radius_spin.value(),
        )

    def _get_prob_path_for_preview(self) -> Path | None:
        input_dir = self._input_edit.text().strip()
        if not input_dir:
            self._status_label.setText("Set cellpose output directory first.")
            return None
        prob_files = discover_prob_files(input_dir)
        idx = self._tp_spin.value()
        if not prob_files:
            self._status_label.setText("No t*_prob.tif files found.")
            return None
        if idx >= len(prob_files):
            self._status_label.setText(f"Only {len(prob_files)} timepoints available.")
            return None
        return prob_files[idx]

    # ── Interactive preview ──────────────────────────────────────────────

    def _schedule_update(self) -> None:
        if self._mag is not None:
            self._update_timer.start()

    def _on_preview(self) -> None:
        prob_path = self._get_prob_path_for_preview()
        if prob_path is None:
            return
        self._status_label.setText(f"Loading {prob_path.name}\u2026")
        self._mag = load_prob_map(prob_path)
        if self._mag_layer is None or self._mag_layer not in self.viewer.layers:
            self._mag_layer = self.viewer.add_image(
                self._mag, name="cell probability", colormap="inferno"
            )
        else:
            self._mag_layer.data = self._mag
        self._update_preview()
        self._edit_btn.setEnabled(True)
        self._status_label.setText(f"Preview: {prob_path.name}")

    def _update_preview(self) -> None:
        if self._mag is None:
            return
        cfg = self._build_config()
        has_preproc = cfg.median_filter or cfg.gaussian_filter or cfg.clahe
        if has_preproc:
            preproc_mag = apply_blur(self._mag, cfg)
            preproc_mag = apply_clahe(preproc_mag, cfg)
            if self._clahe_layer is None or self._clahe_layer not in self.viewer.layers:
                self._clahe_layer = self.viewer.add_image(
                    preproc_mag, name="preprocessed magnitude",
                    colormap="magma", visible=True,
                )
            else:
                self._clahe_layer.data = preproc_mag
        elif self._clahe_layer is not None and self._clahe_layer in self.viewer.layers:
            self.viewer.layers.remove(self._clahe_layer)
            self._clahe_layer = None

        mask = compute_foreground_from_mag(self._mag, cfg)
        if self._preview_layer is None or self._preview_layer not in self.viewer.layers:
            self._preview_layer = self.viewer.add_labels(
                mask.astype(np.int32), name="foreground preview"
            )
        else:
            self._preview_layer.data = mask.astype(np.int32)

    def _on_edit_mask(self) -> None:
        if self._preview_layer is not None and self._preview_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self._preview_layer
            self._preview_layer.mode = "paint"
            self._status_label.setText(
                "Editing mask — use paint (1) / erase (0) tools."
            )

    # ── Run foreground ───────────────────────────────────────────────────

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
            for update in run_s02(input_dir, output_dir, cfg, overwrite=overwrite):
                yield update

        self._worker = _work()

    def _on_run_terminal(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        if not input_dir or not output_dir:
            self._status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_fg_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._overwrite_check.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s02_foreground"
            f" --input-dir \"{input_dir}\""
            f" --output-dir \"{output_dir}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._status_label.setText("Launched foreground stage in terminal.")
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
        self._status_label.setText("Done \u2014 foreground.tif written.")
        self._worker = None
        self._load_foreground_stack()

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None

    # ── Load results ────────────────────────────────────────────────────

    def _load_foreground_stack(self) -> None:
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            return
        fg_path = Path(output_dir) / "foreground.tif"
        if not fg_path.exists():
            self._status_label.setText("No foreground.tif found in output directory.")
            return
        stack = tifffile.imread(str(fg_path))
        self.viewer.add_labels(stack.astype(np.int32), name="foreground (all timepoints)")
        self._status_label.setText(f"Loaded foreground stack: {stack.shape}")

    def _on_load_results(self) -> None:
        self._load_foreground_stack()

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
        cfg = ForegroundConfig(**data)
        self._apply_config(cfg)
        self._status_label.setText(f"Parameters loaded from {Path(path).name}")

    def _apply_config(self, cfg: ForegroundConfig) -> None:
        self._median_check.setChecked(cfg.median_filter)
        self._median_radius_spin.setValue(cfg.median_radius)
        self._gaussian_check.setChecked(cfg.gaussian_filter)
        self._gaussian_sigma_spin.setValue(cfg.gaussian_sigma)
        self._clahe_check.setChecked(cfg.clahe)
        self._clahe_clip_spin.setValue(cfg.clahe_clip_limit)
        self._clahe_kernel_spin.setValue(cfg.clahe_kernel_size)
        self._method_combo.setCurrentText(cfg.method)
        self._thresh_spin.setValue(cfg.threshold)
        self._sigmoid_center_spin.setValue(cfg.sigmoid_center)
        self._sigmoid_steepness_spin.setValue(cfg.sigmoid_steepness)
        self._fill_holes_check.setChecked(cfg.fill_holes)
        self._fill_holes_max_spin.setValue(cfg.fill_holes_max_size)
        self._morpho_combo.setCurrentText(cfg.morpho_op)
        self._morpho_radius_spin.setValue(cfg.morpho_radius)
        self._remove_small_check.setChecked(cfg.remove_small)
        self._remove_small_spin.setValue(cfg.remove_small_min_size)
        self._area_filter_check.setChecked(cfg.area_filter)
        self._area_min_spin.setValue(cfg.area_filter_min)
        self._area_max_spin.setValue(cfg.area_filter_max)
        self._distance_filter_check.setChecked(cfg.distance_filter)
        self._distance_radius_spin.setValue(cfg.distance_filter_min_radius)
