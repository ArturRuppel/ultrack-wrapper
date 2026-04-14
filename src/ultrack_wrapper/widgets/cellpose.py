"""Cellpose segmentation panels (s01a + s01b)."""

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
    QTabWidget,
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
from ultrack_wrapper.stages.s01b_cellpose_cell import run as run_s01b


class CellposeWidget(QWidget):
    """Widget for running Cellpose segmentation (s01a nucleus + s01b cell)."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker_s01a = None
        self._worker_s01b = None

        layout = QVBoxLayout()

        # Create tab widget for s01a and s01b
        self._tabs = QTabWidget()
        self._s01a_widget = self._create_s01a_widget()
        self._s01b_widget = self._create_s01b_widget()
        self._tabs.addTab(self._s01a_widget, "s01a — 3D Nucleus")
        self._tabs.addTab(self._s01b_widget, "s01b — 2D Cell")
        layout.addWidget(self._tabs)

        self.setLayout(layout)

    def _create_s01a_widget(self) -> QWidget:
        """Create the s01a (3D nucleus) panel."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Input directory ──────────────────────────────────────────────
        layout.addWidget(QLabel("Input directory (raw nucleus TIFFs)"))
        row = QHBoxLayout()
        self._s01a_input_edit = QLineEdit()
        self._s01a_input_edit.setPlaceholderText("/path/to/0_raw/nucleus")
        row.addWidget(self._s01a_input_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_s01a_input)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Output directory ─────────────────────────────────────────────
        layout.addWidget(QLabel("Output directory"))
        row = QHBoxLayout()
        self._s01a_output_edit = QLineEdit()
        self._s01a_output_edit.setPlaceholderText("/path/to/1a_cellpose_nucleus")
        row.addWidget(self._s01a_output_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_s01a_output)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Model parameters ─────────────────────────────────────────────
        model_group = QGroupBox("Model parameters")
        mg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self._s01a_model_combo = QComboBox()
        self._s01a_model_combo.addItems(["nuclei", "cyto", "cyto2", "cyto3"])
        self._s01a_model_combo.setEditable(True)
        row.addWidget(self._s01a_model_combo)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Diameter (px, 0=auto)"))
        self._s01a_diameter_spin = QDoubleSpinBox()
        self._s01a_diameter_spin.setRange(0.0, 500.0)
        self._s01a_diameter_spin.setSingleStep(1.0)
        self._s01a_diameter_spin.setDecimals(1)
        self._s01a_diameter_spin.setValue(17.0)
        row.addWidget(self._s01a_diameter_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Anisotropy (Z/XY voxel ratio)"))
        self._s01a_anisotropy_spin = QDoubleSpinBox()
        self._s01a_anisotropy_spin.setRange(0.1, 20.0)
        self._s01a_anisotropy_spin.setSingleStep(0.1)
        self._s01a_anisotropy_spin.setDecimals(2)
        self._s01a_anisotropy_spin.setValue(1.0)
        row.addWidget(self._s01a_anisotropy_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min size (voxels)"))
        self._s01a_min_size_spin = QSpinBox()
        self._s01a_min_size_spin.setRange(0, 10_000_000)
        self._s01a_min_size_spin.setSingleStep(100)
        self._s01a_min_size_spin.setValue(500)
        row.addWidget(self._s01a_min_size_spin)
        mg_layout.addLayout(row)

        self._s01a_use_gpu_check = QCheckBox("Use GPU")
        self._s01a_use_gpu_check.setChecked(True)
        mg_layout.addWidget(self._s01a_use_gpu_check)

        model_group.setLayout(mg_layout)
        layout.addWidget(model_group)

        # ── Preprocessing ────────────────────────────────────────────────
        preproc_group = QGroupBox("Preprocessing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._s01a_gamma_check = QCheckBox("Gamma correction")
        self._s01a_gamma_check.toggled.connect(self._on_s01a_gamma_toggled)
        row.addWidget(self._s01a_gamma_check)
        self._s01a_gamma_spin = QDoubleSpinBox()
        self._s01a_gamma_spin.setRange(0.1, 5.0)
        self._s01a_gamma_spin.setSingleStep(0.1)
        self._s01a_gamma_spin.setDecimals(2)
        self._s01a_gamma_spin.setValue(1.0)
        self._s01a_gamma_spin.setEnabled(False)
        row.addWidget(self._s01a_gamma_spin)
        pp_layout.addLayout(row)

        preproc_group.setLayout(pp_layout)
        layout.addWidget(preproc_group)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._s01a_run_btn = QPushButton("Run Cellpose")
        self._s01a_run_btn.clicked.connect(self._on_s01a_run)
        row.addWidget(self._s01a_run_btn)
        self._s01a_run_terminal_btn = QPushButton("Run in Terminal")
        self._s01a_run_terminal_btn.clicked.connect(self._on_s01a_run_terminal)
        row.addWidget(self._s01a_run_terminal_btn)
        layout.addLayout(row)

        # ── Overwrite + Load results ─────────────────────────────────────
        self._s01a_overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self._s01a_overwrite_check)

        self._s01a_load_results_btn = QPushButton("Load Results")
        self._s01a_load_results_btn.clicked.connect(self._on_s01a_load_results)
        layout.addWidget(self._s01a_load_results_btn)

        # ── Save / Load parameters ───────────────────────────────────────
        row = QHBoxLayout()
        save_btn = QPushButton("Save Parameters\u2026")
        save_btn.clicked.connect(self._on_s01a_save_params)
        row.addWidget(save_btn)
        load_btn = QPushButton("Load Parameters\u2026")
        load_btn.clicked.connect(self._on_s01a_load_params)
        row.addWidget(load_btn)
        layout.addLayout(row)

        # ── Progress ─────────────────────────────────────────────────────
        self._s01a_progress = QProgressBar()
        self._s01a_progress.setVisible(False)
        layout.addWidget(self._s01a_progress)

        self._s01a_status_label = QLabel("")
        layout.addWidget(self._s01a_status_label)

        widget.setLayout(layout)
        return widget

    def _create_s01b_widget(self) -> QWidget:
        """Create the s01b (2D cell) panel."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Root directory ───────────────────────────────────────────────
        layout.addWidget(QLabel("Root project directory"))
        layout.addWidget(QLabel("<small>Parent directory of pos00, pos01, etc.</small>"))
        row = QHBoxLayout()
        self._s01b_root_edit = QLineEdit()
        self._s01b_root_edit.setPlaceholderText("/path/to/project (will auto-detect if you select a subdirectory)")
        row.addWidget(self._s01b_root_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_s01b_root)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Position ─────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Position"))
        self._s01b_pos_spin = QSpinBox()
        self._s01b_pos_spin.setRange(0, 1000)
        self._s01b_pos_spin.setValue(0)
        row.addWidget(self._s01b_pos_spin)
        layout.addLayout(row)

        # ── Model parameters ─────────────────────────────────────────────
        model_group = QGroupBox("Model parameters")
        mg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self._s01b_model_combo = QComboBox()
        self._s01b_model_combo.addItems(["cyto3", "cyto2", "cyto", "nuclei"])
        self._s01b_model_combo.setEditable(True)
        self._s01b_model_combo.setCurrentText("cyto3")
        row.addWidget(self._s01b_model_combo)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Diameter (px, 0=auto)"))
        self._s01b_diameter_spin = QDoubleSpinBox()
        self._s01b_diameter_spin.setRange(0.0, 500.0)
        self._s01b_diameter_spin.setSingleStep(1.0)
        self._s01b_diameter_spin.setDecimals(1)
        self._s01b_diameter_spin.setValue(30.0)
        row.addWidget(self._s01b_diameter_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min size (voxels)"))
        self._s01b_min_size_spin = QSpinBox()
        self._s01b_min_size_spin.setRange(0, 10_000_000)
        self._s01b_min_size_spin.setSingleStep(100)
        self._s01b_min_size_spin.setValue(200)
        row.addWidget(self._s01b_min_size_spin)
        mg_layout.addLayout(row)

        self._s01b_use_gpu_check = QCheckBox("Use GPU")
        self._s01b_use_gpu_check.setChecked(True)
        mg_layout.addWidget(self._s01b_use_gpu_check)

        model_group.setLayout(mg_layout)
        layout.addWidget(model_group)

        # ── Preprocessing ────────────────────────────────────────────────
        preproc_group = QGroupBox("Preprocessing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._s01b_gamma_check = QCheckBox("Gamma correction")
        self._s01b_gamma_check.toggled.connect(self._on_s01b_gamma_toggled)
        row.addWidget(self._s01b_gamma_check)
        self._s01b_gamma_spin = QDoubleSpinBox()
        self._s01b_gamma_spin.setRange(0.1, 5.0)
        self._s01b_gamma_spin.setSingleStep(0.1)
        self._s01b_gamma_spin.setDecimals(2)
        self._s01b_gamma_spin.setValue(1.0)
        self._s01b_gamma_spin.setEnabled(False)
        row.addWidget(self._s01b_gamma_spin)
        pp_layout.addLayout(row)

        preproc_group.setLayout(pp_layout)
        layout.addWidget(preproc_group)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._s01b_run_btn = QPushButton("Run Cellpose")
        self._s01b_run_btn.clicked.connect(self._on_s01b_run)
        row.addWidget(self._s01b_run_btn)
        layout.addLayout(row)

        # ── Overwrite + Load results ─────────────────────────────────────
        self._s01b_overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self._s01b_overwrite_check)

        self._s01b_load_results_btn = QPushButton("Load Results")
        self._s01b_load_results_btn.clicked.connect(self._on_s01b_load_results)
        layout.addWidget(self._s01b_load_results_btn)

        # ── Save / Load parameters ───────────────────────────────────────
        row = QHBoxLayout()
        save_btn = QPushButton("Save Parameters\u2026")
        save_btn.clicked.connect(self._on_s01b_save_params)
        row.addWidget(save_btn)
        load_btn = QPushButton("Load Parameters\u2026")
        load_btn.clicked.connect(self._on_s01b_load_params)
        row.addWidget(load_btn)
        layout.addLayout(row)

        # ── Progress ─────────────────────────────────────────────────────
        self._s01b_progress = QProgressBar()
        self._s01b_progress.setVisible(False)
        layout.addWidget(self._s01b_progress)

        self._s01b_status_label = QLabel("")
        layout.addWidget(self._s01b_status_label)

        widget.setLayout(layout)
        return widget

    # ── s01a Helpers ─────────────────────────────────────────────────────

    def _browse_s01a_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select raw nucleus TIFF directory")
        if d:
            self._s01a_input_edit.setText(d)

    def _browse_s01a_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select cellpose output directory")
        if d:
            self._s01a_output_edit.setText(d)

    # ── s01b Helpers ─────────────────────────────────────────────────────

    def _browse_s01b_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Select root project directory (parent of pos00, pos01, etc.)"
        )
        if d:
            # Auto-detect: if user selected a pos## or stage directory, go up
            d_path = Path(d)
            if d_path.name.startswith("pos") and d_path.name[3:].isdigit():
                d = str(d_path.parent)
            elif d_path.name in ["0_raw", "1a_cellpose_nucleus", "1b_cellpose_cell"]:
                d = str(d_path.parent.parent)
            self._s01b_root_edit.setText(d)

    def _on_s01a_gamma_toggled(self, checked: bool) -> None:
        self._s01a_gamma_spin.setEnabled(checked)

    def _on_s01b_gamma_toggled(self, checked: bool) -> None:
        self._s01b_gamma_spin.setEnabled(checked)

    def _build_s01a_config(self) -> CellposeConfig:
        gamma = self._s01a_gamma_spin.value() if self._s01a_gamma_check.isChecked() else None
        return CellposeConfig(
            model=self._s01a_model_combo.currentText(),
            diameter=self._s01a_diameter_spin.value(),
            anisotropy=self._s01a_anisotropy_spin.value(),
            min_size=self._s01a_min_size_spin.value(),
            use_gpu=self._s01a_use_gpu_check.isChecked(),
            gamma=gamma,
        )

    def _build_s01b_config(self) -> CellposeConfig:
        gamma = self._s01b_gamma_spin.value() if self._s01b_gamma_check.isChecked() else None
        return CellposeConfig(
            model=self._s01b_model_combo.currentText(),
            diameter=self._s01b_diameter_spin.value(),
            anisotropy=0.0,  # Not used for 2D
            min_size=self._s01b_min_size_spin.value(),
            use_gpu=self._s01b_use_gpu_check.isChecked(),
            gamma=gamma,
        )

    def _apply_s01a_config(self, cfg: CellposeConfig) -> None:
        self._s01a_model_combo.setCurrentText(cfg.model)
        self._s01a_diameter_spin.setValue(cfg.diameter)
        self._s01a_anisotropy_spin.setValue(cfg.anisotropy)
        self._s01a_min_size_spin.setValue(cfg.min_size)
        self._s01a_use_gpu_check.setChecked(cfg.use_gpu)
        if cfg.gamma is not None:
            self._s01a_gamma_check.setChecked(True)
            self._s01a_gamma_spin.setValue(cfg.gamma)
        else:
            self._s01a_gamma_check.setChecked(False)

    def _apply_s01b_config(self, cfg: CellposeConfig) -> None:
        self._s01b_model_combo.setCurrentText(cfg.model)
        self._s01b_diameter_spin.setValue(cfg.diameter)
        self._s01b_min_size_spin.setValue(cfg.min_size)
        self._s01b_use_gpu_check.setChecked(cfg.use_gpu)
        if cfg.gamma is not None:
            self._s01b_gamma_check.setChecked(True)
            self._s01b_gamma_spin.setValue(cfg.gamma)
        else:
            self._s01b_gamma_check.setChecked(False)

    # ── s01a Run inline ──────────────────────────────────────────────────

    def _on_s01a_run(self) -> None:
        input_dir = self._s01a_input_edit.text().strip()
        output_dir = self._s01a_output_edit.text().strip()
        if not input_dir or not output_dir:
            self._s01a_status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_s01a_config()
        overwrite = self._s01a_overwrite_check.isChecked()
        self._s01a_run_btn.setEnabled(False)
        self._s01a_run_terminal_btn.setEnabled(False)
        self._s01a_progress.setVisible(True)
        self._s01a_progress.setValue(0)
        self._s01a_status_label.setText("Starting\u2026")

        @thread_worker(
            connect={
                "yielded": self._on_s01a_progress,
                "finished": self._on_s01a_finished,
                "errored": self._on_s01a_error,
            }
        )
        def _work():
            for update in run_s01a(input_dir, output_dir, cfg, overwrite=overwrite):
                yield update

        self._worker_s01a = _work()

    # ── s01b Run inline ──────────────────────────────────────────────────

    def _on_s01b_run(self) -> None:
        root_dir = self._s01b_root_edit.text().strip()
        if not root_dir:
            self._s01b_status_label.setText("Set root project directory.")
            return
        pos = self._s01b_pos_spin.value()
        cfg = self._build_s01b_config()
        overwrite = self._s01b_overwrite_check.isChecked()
        self._s01b_run_btn.setEnabled(False)
        self._s01b_progress.setVisible(True)
        self._s01b_progress.setValue(0)
        self._s01b_status_label.setText("Starting\u2026")

        @thread_worker(
            connect={
                "yielded": self._on_s01b_progress,
                "finished": self._on_s01b_finished,
                "errored": self._on_s01b_error,
            }
        )
        def _work():
            for update in run_s01b(root_dir, pos, cfg, overwrite=overwrite):
                yield update

        self._worker_s01b = _work()

    # ── s01a Run in terminal ─────────────────────────────────────────────

    def _on_s01a_run_terminal(self) -> None:
        input_dir = self._s01a_input_edit.text().strip()
        output_dir = self._s01a_output_edit.text().strip()
        if not input_dir or not output_dir:
            self._s01a_status_label.setText("Set both input and output directories.")
            return
        cfg = self._build_s01a_config()
        cfg_path = Path(tempfile.mktemp(suffix="_cp_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._s01a_overwrite_check.isChecked() else ""
        cmd = (
            f"python -m ultrack_wrapper.stages.s01a_cellpose_nucleus"
            f" --input-dir \"{input_dir}\""
            f" --output-dir \"{output_dir}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._s01a_status_label.setText("Launched Cellpose stage in terminal.")
        except Exception as e:
            self._s01a_status_label.setText(f"Terminal launch error: {e}")

    # ── s01a Progress / finished / error callbacks ────────────────────────

    def _on_s01a_progress(self, update: tuple) -> None:
        done, total, label = update
        self._s01a_progress.setMaximum(max(total, 1))
        self._s01a_progress.setValue(done)
        self._s01a_status_label.setText(f"Processing {label} [{done}/{total}]")

    def _on_s01a_finished(self) -> None:
        self._s01a_run_btn.setEnabled(True)
        self._s01a_run_terminal_btn.setEnabled(True)
        self._s01a_progress.setVisible(False)
        self._s01a_status_label.setText("Done \u2014 Cellpose outputs written.")
        self._worker_s01a = None
        self._load_s01a_prob_stack()

    def _on_s01a_error(self, exc: Exception) -> None:
        self._s01a_run_btn.setEnabled(True)
        self._s01a_run_terminal_btn.setEnabled(True)
        self._s01a_progress.setVisible(False)
        self._s01a_status_label.setText(f"Error: {exc}")
        self._worker_s01a = None

    # ── s01b Progress / finished / error callbacks ────────────────────────

    def _on_s01b_progress(self, update: tuple) -> None:
        done, total, label = update
        self._s01b_progress.setMaximum(max(total, 1))
        self._s01b_progress.setValue(done)
        self._s01b_status_label.setText(f"Processing t{label:03d} [{done}/{total}]")

    def _on_s01b_finished(self) -> None:
        self._s01b_run_btn.setEnabled(True)
        self._s01b_progress.setVisible(False)
        self._s01b_status_label.setText("Done \u2014 Cellpose outputs written.")
        self._worker_s01b = None
        self._load_s01b_results()

    def _on_s01b_error(self, exc: Exception) -> None:
        self._s01b_run_btn.setEnabled(True)
        self._s01b_progress.setVisible(False)
        self._s01b_status_label.setText(f"Error: {exc}")
        self._worker_s01b = None

    # ── s01a Load results ────────────────────────────────────────────────

    def _load_s01a_prob_stack(self) -> None:
        output_dir = self._s01a_output_edit.text().strip()
        if not output_dir:
            return
        prob_files = sorted(Path(output_dir).glob("*_prob.tif"))
        if not prob_files:
            return
        # Load first prob file as a preview
        prob = tifffile.imread(str(prob_files[0]))
        self.viewer.add_image(prob, name="nucleus_cellprob (t0)", colormap="inferno")
        self._s01a_status_label.setText(
            f"Loaded {prob_files[0].name}  shape={prob.shape}"
        )

    def _on_s01a_load_results(self) -> None:
        output_dir = self._s01a_output_edit.text().strip()
        if not output_dir:
            self._s01a_status_label.setText("Set output directory first.")
            return
        prob_files = sorted(Path(output_dir).glob("*_prob.tif"))
        if not prob_files:
            self._s01a_status_label.setText("No *_prob.tif files found in output directory.")
            return
        for pf in prob_files:
            prob = tifffile.imread(str(pf))
            self.viewer.add_image(prob, name=pf.stem, colormap="inferno")
        self._s01a_status_label.setText(f"Loaded {len(prob_files)} probability map(s).")

    # ── s01b Load results ────────────────────────────────────────────────

    def _load_s01b_results(self) -> None:
        root_dir = self._s01b_root_edit.text().strip()
        if not root_dir:
            return
        pos = self._s01b_pos_spin.value()
        from ultrack_wrapper._paths import cellpose_cell_dir

        cell_dir = cellpose_cell_dir(root_dir, pos)
        prob_file = cell_dir / "cell_prob.tif"
        if not prob_file.exists():
            return
        prob = tifffile.imread(str(prob_file))
        self.viewer.add_image(prob, name=f"cell_prob_pos{pos:02d}", colormap="inferno")
        self._s01b_status_label.setText(f"Loaded cell_prob.tif  shape={prob.shape}")

    def _on_s01b_load_results(self) -> None:
        root_dir = self._s01b_root_edit.text().strip()
        if not root_dir:
            self._s01b_status_label.setText("Set root directory first.")
            return
        pos = self._s01b_pos_spin.value()
        from ultrack_wrapper._paths import cellpose_cell_dir

        cell_dir = cellpose_cell_dir(root_dir, pos)
        prob_file = cell_dir / "cell_prob.tif"
        dp_file = cell_dir / "cell_dp.tif"

        if prob_file.exists():
            prob = tifffile.imread(str(prob_file))
            self.viewer.add_image(prob, name=f"cell_prob_pos{pos:02d}", colormap="inferno")

        if dp_file.exists():
            dp = tifffile.imread(str(dp_file))
            self.viewer.add_image(dp, name=f"cell_dp_pos{pos:02d}", colormap="viridis")

        if prob_file.exists() or dp_file.exists():
            self._s01b_status_label.setText(f"Loaded cellpose outputs for pos{pos:02d}")
        else:
            self._s01b_status_label.setText("No cellpose outputs found.")

    # ── s01a Save / Load parameters ──────────────────────────────────────

    def _on_s01a_save_params(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save s01a parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        cfg = self._build_s01a_config()
        Path(path).write_text(json.dumps(cfg.model_dump(), indent=2))
        self._s01a_status_label.setText(f"Parameters saved to {Path(path).name}")

    def _on_s01a_load_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load s01a parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        cfg = CellposeConfig(**data)
        self._apply_s01a_config(cfg)
        self._s01a_status_label.setText(f"Parameters loaded from {Path(path).name}")

    # ── s01b Save / Load parameters ──────────────────────────────────────

    def _on_s01b_save_params(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save s01b parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        cfg = self._build_s01b_config()
        Path(path).write_text(json.dumps(cfg.model_dump(), indent=2))
        self._s01b_status_label.setText(f"Parameters saved to {Path(path).name}")

    def _on_s01b_load_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load s01b parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        cfg = CellposeConfig(**data)
        self._apply_s01b_config(cfg)
        self._s01b_status_label.setText(f"Parameters loaded from {Path(path).name}")
