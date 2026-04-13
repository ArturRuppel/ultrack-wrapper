"""Tracking panel — s03 Ultrack segment + link + solve."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
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

from ultrack_wrapper._config import TrackingConfig
from ultrack_wrapper.runners.terminal import launch_in_terminal
from ultrack_wrapper.stages.s03_tracking import (
    export_ctc,
    get_labels_layer,
    get_tracks_layer,
    run as run_s03,
)


class TrackingWidget(QWidget):
    """Ultrack tracking: consumes foreground + contour maps."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── Input: foreground stack ──────────────────────────────────────
        layout.addWidget(QLabel("Foreground stack (foreground.tif)"))
        row = QHBoxLayout()
        self._fg_edit = QLineEdit()
        self._fg_edit.setPlaceholderText("/path/to/2_foreground/foreground.tif")
        row.addWidget(self._fg_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_fg)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Input: contours stack ────────────────────────────────────────
        layout.addWidget(QLabel("Contours stack (contours.tif)"))
        row = QHBoxLayout()
        self._contours_edit = QLineEdit()
        self._contours_edit.setPlaceholderText("/path/to/2b_contours/contours.tif")
        row.addWidget(self._contours_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_contours)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Working directory ────────────────────────────────────────────
        layout.addWidget(QLabel("Working directory (Ultrack database)"))
        row = QHBoxLayout()
        self._wd_edit = QLineEdit()
        self._wd_edit.setPlaceholderText("/path/to/3_tracking")
        row.addWidget(self._wd_edit)
        btn = QPushButton("Browse\u2026")
        btn.clicked.connect(self._browse_wd)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Segmentation group ───────────────────────────────────────────
        seg_group = QGroupBox("Segmentation hypotheses")
        seg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Min area"))
        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(1, 10_000_000)
        self._min_area_spin.setValue(100)
        row.addWidget(self._min_area_spin)
        row.addWidget(QLabel("Max area"))
        self._max_area_spin = QSpinBox()
        self._max_area_spin.setRange(1, 10_000_000)
        self._max_area_spin.setValue(1_000_000)
        row.addWidget(self._max_area_spin)
        seg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min frontier"))
        self._min_frontier_spin = QDoubleSpinBox()
        self._min_frontier_spin.setRange(0.0, 1.0)
        self._min_frontier_spin.setSingleStep(0.05)
        self._min_frontier_spin.setDecimals(3)
        self._min_frontier_spin.setValue(0.0)
        row.addWidget(self._min_frontier_spin)
        row.addWidget(QLabel("FG threshold"))
        self._seg_thresh_spin = QDoubleSpinBox()
        self._seg_thresh_spin.setRange(0.0, 1.0)
        self._seg_thresh_spin.setSingleStep(0.05)
        self._seg_thresh_spin.setDecimals(2)
        self._seg_thresh_spin.setValue(0.5)
        row.addWidget(self._seg_thresh_spin)
        seg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("WS hierarchy"))
        self._ws_combo = QComboBox()
        self._ws_combo.addItems(["area", "dynamics", "volume"])
        row.addWidget(self._ws_combo)
        row.addWidget(QLabel("Aniso. pen."))
        self._aniso_spin = QDoubleSpinBox()
        self._aniso_spin.setRange(-10.0, 10.0)
        self._aniso_spin.setSingleStep(0.1)
        self._aniso_spin.setDecimals(1)
        self._aniso_spin.setValue(0.0)
        row.addWidget(self._aniso_spin)
        seg_layout.addLayout(row)

        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

        # ── Linking group ────────────────────────────────────────────────
        link_group = QGroupBox("Linking")
        link_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Max distance"))
        self._max_dist_spin = QDoubleSpinBox()
        self._max_dist_spin.setRange(0.1, 500.0)
        self._max_dist_spin.setSingleStep(1.0)
        self._max_dist_spin.setDecimals(1)
        self._max_dist_spin.setValue(15.0)
        row.addWidget(self._max_dist_spin)
        row.addWidget(QLabel("Max neighbors"))
        self._max_neighbors_spin = QSpinBox()
        self._max_neighbors_spin.setRange(1, 50)
        self._max_neighbors_spin.setValue(5)
        row.addWidget(self._max_neighbors_spin)
        link_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Distance weight"))
        self._dist_weight_spin = QDoubleSpinBox()
        self._dist_weight_spin.setRange(0.0, 10.0)
        self._dist_weight_spin.setSingleStep(0.1)
        self._dist_weight_spin.setDecimals(2)
        self._dist_weight_spin.setValue(0.0)
        row.addWidget(self._dist_weight_spin)
        link_layout.addLayout(row)

        link_group.setLayout(link_layout)
        layout.addWidget(link_group)

        # ── Solver / ILP group ───────────────────────────────────────────
        solver_group = QGroupBox("Solver (ILP)")
        solver_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Appear"))
        self._appear_spin = QDoubleSpinBox()
        self._appear_spin.setRange(-100.0, 0.0)
        self._appear_spin.setSingleStep(0.001)
        self._appear_spin.setDecimals(4)
        self._appear_spin.setValue(-0.001)
        row.addWidget(self._appear_spin)
        row.addWidget(QLabel("Disappear"))
        self._disappear_spin = QDoubleSpinBox()
        self._disappear_spin.setRange(-100.0, 0.0)
        self._disappear_spin.setSingleStep(0.001)
        self._disappear_spin.setDecimals(4)
        self._disappear_spin.setValue(-0.001)
        row.addWidget(self._disappear_spin)
        solver_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Division"))
        self._division_spin = QDoubleSpinBox()
        self._division_spin.setRange(-100.0, 0.0)
        self._division_spin.setSingleStep(0.001)
        self._division_spin.setDecimals(4)
        self._division_spin.setValue(-0.001)
        row.addWidget(self._division_spin)
        row.addWidget(QLabel("Link func"))
        self._link_func_combo = QComboBox()
        self._link_func_combo.addItems(["power", "identity"])
        row.addWidget(self._link_func_combo)
        solver_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Power"))
        self._power_spin = QDoubleSpinBox()
        self._power_spin.setRange(0.1, 20.0)
        self._power_spin.setSingleStep(0.5)
        self._power_spin.setDecimals(1)
        self._power_spin.setValue(4.0)
        row.addWidget(self._power_spin)
        row.addWidget(QLabel("Bias"))
        self._bias_spin = QDoubleSpinBox()
        self._bias_spin.setRange(-10.0, 0.0)
        self._bias_spin.setSingleStep(0.01)
        self._bias_spin.setDecimals(3)
        self._bias_spin.setValue(0.0)
        row.addWidget(self._bias_spin)
        solver_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gap"))
        self._gap_spin = QDoubleSpinBox()
        self._gap_spin.setRange(0.0, 1.0)
        self._gap_spin.setSingleStep(0.001)
        self._gap_spin.setDecimals(4)
        self._gap_spin.setValue(0.001)
        row.addWidget(self._gap_spin)
        row.addWidget(QLabel("Time limit (s)"))
        self._time_limit_spin = QSpinBox()
        self._time_limit_spin.setRange(10, 360_000)
        self._time_limit_spin.setValue(36_000)
        row.addWidget(self._time_limit_spin)
        solver_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Window size (0=all)"))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(0, 10_000)
        self._window_spin.setValue(0)
        row.addWidget(self._window_spin)
        row.addWidget(QLabel("Workers"))
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 64)
        self._workers_spin.setValue(1)
        row.addWidget(self._workers_spin)
        solver_layout.addLayout(row)

        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)

        # ── Overwrite mode ───────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Overwrite"))
        self._overwrite_combo = QComboBox()
        self._overwrite_combo.addItems(["all", "links", "solutions", "none"])
        self._overwrite_combo.setCurrentText("all")
        row.addWidget(self._overwrite_combo)
        layout.addLayout(row)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._run_btn)
        self._run_terminal_btn = QPushButton("Run in Terminal")
        self._run_terminal_btn.clicked.connect(self._on_run_terminal)
        row.addWidget(self._run_terminal_btn)
        layout.addLayout(row)

        # ── Load / Export ────────────────────────────────────────────────
        self._load_btn = QPushButton("Load Results into Viewer")
        self._load_btn.clicked.connect(self._on_load_results)
        layout.addWidget(self._load_btn)

        self._export_ctc_btn = QPushButton("Export CTC\u2026")
        self._export_ctc_btn.clicked.connect(self._on_export_ctc)
        layout.addWidget(self._export_ctc_btn)

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

    # ── Browse helpers ──────────────────────────────────────────────────

    def _browse_fg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select foreground stack", "", "TIFF files (*.tif *.tiff)"
        )
        if path:
            self._fg_edit.setText(path)

    def _browse_contours(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select contours stack", "", "TIFF files (*.tif *.tiff)"
        )
        if path:
            self._contours_edit.setText(path)

    def _browse_wd(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select working directory")
        if d:
            self._wd_edit.setText(d)

    # ── Config ──────────────────────────────────────────────────────────

    def _build_config(self) -> TrackingConfig:
        return TrackingConfig(
            min_area=self._min_area_spin.value(),
            max_area=self._max_area_spin.value(),
            min_frontier=self._min_frontier_spin.value(),
            threshold=self._seg_thresh_spin.value(),
            ws_hierarchy=self._ws_combo.currentText(),
            anisotropy_penalization=self._aniso_spin.value(),
            n_workers=self._workers_spin.value(),
            max_distance=self._max_dist_spin.value(),
            max_neighbors=self._max_neighbors_spin.value(),
            distance_weight=self._dist_weight_spin.value(),
            appear_weight=self._appear_spin.value(),
            disappear_weight=self._disappear_spin.value(),
            division_weight=self._division_spin.value(),
            link_function=self._link_func_combo.currentText(),
            power=self._power_spin.value(),
            bias=self._bias_spin.value(),
            solution_gap=self._gap_spin.value(),
            time_limit=self._time_limit_spin.value(),
            window_size=self._window_spin.value(),
            overwrite=self._overwrite_combo.currentText(),
        )

    def _apply_config(self, cfg: TrackingConfig) -> None:
        self._min_area_spin.setValue(cfg.min_area)
        self._max_area_spin.setValue(cfg.max_area)
        self._min_frontier_spin.setValue(cfg.min_frontier)
        self._seg_thresh_spin.setValue(cfg.threshold)
        self._ws_combo.setCurrentText(cfg.ws_hierarchy)
        self._aniso_spin.setValue(cfg.anisotropy_penalization)
        self._workers_spin.setValue(cfg.n_workers)
        self._max_dist_spin.setValue(cfg.max_distance)
        self._max_neighbors_spin.setValue(cfg.max_neighbors)
        self._dist_weight_spin.setValue(cfg.distance_weight)
        self._appear_spin.setValue(cfg.appear_weight)
        self._disappear_spin.setValue(cfg.disappear_weight)
        self._division_spin.setValue(cfg.division_weight)
        self._link_func_combo.setCurrentText(cfg.link_function)
        self._power_spin.setValue(cfg.power)
        self._bias_spin.setValue(cfg.bias)
        self._gap_spin.setValue(cfg.solution_gap)
        self._time_limit_spin.setValue(cfg.time_limit)
        self._window_spin.setValue(cfg.window_size)
        self._overwrite_combo.setCurrentText(cfg.overwrite)

    # ── Run tracking ────────────────────────────────────────────────────

    def _on_run(self) -> None:
        fg_path = self._fg_edit.text().strip()
        contours_path = self._contours_edit.text().strip()
        wd = self._wd_edit.text().strip()
        if not fg_path or not contours_path or not wd:
            self._status_label.setText("Set foreground, contours, and working directory.")
            return
        cfg = self._build_config()
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
            for update in run_s03(fg_path, contours_path, wd, cfg):
                yield update

        self._worker = _work()

    def _on_run_terminal(self) -> None:
        fg_path = self._fg_edit.text().strip()
        contours_path = self._contours_edit.text().strip()
        wd = self._wd_edit.text().strip()
        if not fg_path or not contours_path or not wd:
            self._status_label.setText("Set foreground, contours, and working directory.")
            return
        cfg = self._build_config()
        cfg_path = Path(tempfile.mktemp(suffix="_tracking_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        cmd = (
            f"python -m ultrack_wrapper.stages.s03_tracking"
            f" --foreground \"{fg_path}\""
            f" --contours \"{contours_path}\""
            f" --working-dir \"{wd}\""
            f" --config \"{cfg_path}\""
            f" --overwrite \"{cfg.overwrite}\""
        )
        try:
            launch_in_terminal(cmd)
            self._status_label.setText("Launched tracking stage in terminal.")
        except Exception as e:
            self._status_label.setText(f"Terminal launch error: {e}")

    def _on_progress(self, update: tuple) -> None:
        done, total, label = update
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._status_label.setText(label)

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("Tracking complete \u2014 loading results into viewer\u2026")
        self._worker = None
        self._load_results_into_viewer()

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None

    # ── Load results (tracks + labels) ──────────────────────────────────

    def _load_results_into_viewer(self) -> None:
        """Load tracks layer and tracked_labels layer from disk into napari."""
        wd = self._wd_edit.text().strip()
        if not wd:
            self._status_label.setText("Set working directory first.")
            return

        cfg = self._build_config()
        msgs: list[str] = []

        # Tracks layer
        try:
            tracks_df, graph = get_tracks_layer(wd, cfg)
            data = tracks_df.values
            self.viewer.add_tracks(data, graph=graph, name="ultrack tracks")
            msgs.append(
                f"{tracks_df.iloc[:, 0].nunique()} tracks"
                f" ({len(tracks_df)} points)"
            )
        except Exception as e:
            msgs.append(f"tracks error: {e}")

        # Tracked labels layer
        try:
            labels = get_labels_layer(wd)
            self.viewer.add_labels(labels, name="tracked labels")
            msgs.append(f"labels {labels.shape}")
        except FileNotFoundError:
            msgs.append("tracked_labels.tif not found")
        except Exception as e:
            msgs.append(f"labels error: {e}")

        self._status_label.setText("Loaded: " + " | ".join(msgs))

    def _on_load_results(self) -> None:
        self._load_results_into_viewer()

    # ── Export CTC ──────────────────────────────────────────────────────

    def _on_export_ctc(self) -> None:
        wd = self._wd_edit.text().strip()
        if not wd:
            self._status_label.setText("Set working directory first.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select CTC output directory")
        if not output_dir:
            return
        cfg = self._build_config()
        try:
            export_ctc(wd, output_dir, cfg)
            self._status_label.setText(f"CTC export written to {output_dir}")
        except Exception as e:
            self._status_label.setText(f"CTC export error: {e}")

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
        cfg = TrackingConfig(**data)
        self._apply_config(cfg)
        self._status_label.setText(f"Parameters loaded from {Path(path).name}")
