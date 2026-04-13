"""Cellpose 3-D nuclear segmentation panel (s01a)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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

from ultrack_wrapper._config import CellposeConfig
from ultrack_wrapper.runners.terminal import launch_in_terminal
from ultrack_wrapper.stages.s01a_cellpose_nucleus import (
    discover_input_files,
    run as run_s01a,
)


class CellposeWidget(QWidget):
    """Widget for running Cellpose 3-D nuclear segmentation."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Input directory ──────────────────────────────────────────────
        layout.addWidget(QLabel("Input directory (raw nucleus TIFFs)"))
        row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("/path/to/0_raw/nucleus")
        row.addWidget(self._input_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_input)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Output directory ─────────────────────────────────────────────
        layout.addWidget(QLabel("Output directory"))
        row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("/path/to/1a_cellpose_nucleus")
        row.addWidget(self._output_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_output)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Model parameters ─────────────────────────────────────────────
        model_group = QGroupBox("Model parameters")
        mg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["nuclei", "cyto", "cyto2", "cyto3"])
        self._model_combo.setEditable(True)
        row.addWidget(self._model_combo)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Diameter (px, 0=auto)"))
        self._diameter_spin = QDoubleSpinBox()
        self._diameter_spin.setRange(0.0, 500.0)
        self._diameter_spin.setSingleStep(1.0)
        self._diameter_spin.setDecimals(1)
        self._diameter_spin.setValue(17.0)
        row.addWidget(self._diameter_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Anisotropy (Z/XY voxel ratio)"))
        self._anisotropy_spin = QDoubleSpinBox()
        self._anisotropy_spin.setRange(0.1, 20.0)
        self._anisotropy_spin.setSingleStep(0.1)
        self._anisotropy_spin.setDecimals(2)
        self._anisotropy_spin.setValue(1.0)
        row.addWidget(self._anisotropy_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min size (voxels)"))
        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(0, 10_000_000)
        self._min_size_spin.setSingleStep(100)
        self._min_size_spin.setValue(500)
        row.addWidget(self._min_size_spin)
        mg_layout.addLayout(row)

        self._use_gpu_check = QCheckBox("Use GPU")
        self._use_gpu_check.setChecked(True)
        mg_layout.addWidget(self._use_gpu_check)

        model_group.setLayout(mg_layout)
        layout.addWidget(model_group)

        # ── Preprocessing ────────────────────────────────────────────────
        preproc_group = QGroupBox("Preprocessing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._gamma_check = QCheckBox("Gamma correction")
        self._gamma_check.toggled.connect(self._on_gamma_toggled)
        row.addWidget(self._gamma_check)
        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 5.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setValue(1.0)
        self._gamma_spin.setEnabled(False)
        row.addWidget(self._gamma_spin)
        pp_layout.addLayout(row)

        preproc_group.setLayout(pp_layout)
        layout.addWidget(preproc_group)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_btn = QPushButton("Run Cellpose")
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

        self.setLayout(layout)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select raw nucleus TIFF directory")
        if d:
            self._input_edit.setText(d)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select cellpose output directory")
        if d:
            self._output_edit.setText(d)

    def _on_gamma_toggled(self, checked: bool) -> None:
        self._gamma_spin.setEnabled(checked)

    def _build_config(self) -> CellposeConfig:
        gamma = self._gamma_spin.value() if self._gamma_check.isChecked() else None
        return CellposeConfig(
            model=self._model_combo.currentText(),
            diameter=self._diameter_spin.value(),
            anisotropy=self._anisotropy_spin.value(),
            min_size=self._min_size_spin.value(),
            use_gpu=self._use_gpu_check.isChecked(),
            gamma=gamma,
        )

    def _apply_config(self, cfg: CellposeConfig) -> None:
        self._model_combo.setCurrentText(cfg.model)
        self._diameter_spin.setValue(cfg.diameter)
        self._anisotropy_spin.setValue(cfg.anisotropy)
        self._min_size_spin.setValue(cfg.min_size)
        self._use_gpu_check.setChecked(cfg.use_gpu)
        if cfg.gamma is not None:
            self._gamma_check.setChecked(True)
            self._gamma_spin.setValue(cfg.gamma)
        else:
            self._gamma_check.setChecked(False)

    # ── Run inline ───────────────────────────────────────────────────────

    def _on_run(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        if not input_dir or not output_dir:
            self._status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_config()
        overwrite = self._overwrite_check.isChecked()
        self._run_btn.setEnabled(False)
        self._run_terminal_btn.setEnabled(False)
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
            for update in run_s01a(input_dir, output_dir, cfg, overwrite=overwrite):
                yield update

        self._worker = _work()

    # ── Run in terminal ──────────────────────────────────────────────────

    def _on_run_terminal(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        if not input_dir or not output_dir:
            self._status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_cp_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._overwrite_check.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s01a_cellpose_nucleus"
            f" --input-dir \"{input_dir}\""
            f" --output-dir \"{output_dir}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._status_label.setText("Launched Cellpose stage in terminal.")
        except Exception as e:
            self._status_label.setText(f"Terminal launch error: {e}")

    # ── Progress / finished / error callbacks ────────────────────────────

    def _on_progress(self, update: tuple) -> None:
        done, total, label = update
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._status_label.setText(f"Processing {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._run_terminal_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("Done \u2014 Cellpose outputs written.")
        self._worker = None
        self._load_prob_stack()

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._run_terminal_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None

    # ── Load results ─────────────────────────────────────────────────────

    def _load_prob_stack(self) -> None:
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            return
        prob_files = sorted(Path(output_dir).glob("*_prob.tif"))
        if not prob_files:
            return
        # Load first prob file as a preview
        prob = tifffile.imread(str(prob_files[0]))
        self.viewer.add_image(prob, name="cellprob (t0)", colormap="inferno")
        self._status_label.setText(
            f"Loaded {prob_files[0].name}  shape={prob.shape}"
        )

    def _on_load_results(self) -> None:
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            self._status_label.setText("Set output directory first.")
            return
        prob_files = sorted(Path(output_dir).glob("*_prob.tif"))
        if not prob_files:
            self._status_label.setText("No *_prob.tif files found in output directory.")
            return
        for pf in prob_files:
            prob = tifffile.imread(str(pf))
            self.viewer.add_image(prob, name=pf.stem, colormap="inferno")
        self._status_label.setText(f"Loaded {len(prob_files)} probability map(s).")

    # ── Save / Load parameters ───────────────────────────────────────────

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
        cfg = CellposeConfig(**data)
        self._apply_config(cfg)
        self._status_label.setText(f"Parameters loaded from {Path(path).name}")
