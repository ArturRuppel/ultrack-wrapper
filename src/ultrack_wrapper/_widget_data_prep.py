"""Data Preparation tab — s00 raw NDTiff export."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
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

from ultrack_wrapper._config import DatasetConfig
from ultrack_wrapper.stages.s00_raw import run as run_s00


class DataPrepWidget(QWidget):
    """Widget for exporting raw NDTiff data to per-timepoint TIFFs."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── NDTiff path ──────────────────────────────────────────────────
        layout.addWidget(QLabel("NDTiff directory"))
        row = QHBoxLayout()
        self._ndtiff_edit = QLineEdit()
        self._ndtiff_edit.setPlaceholderText("/path/to/ndtiff_dataset")
        row.addWidget(self._ndtiff_edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_ndtiff)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Output root directory ────────────────────────────────────────
        layout.addWidget(QLabel("Output root directory"))
        row = QHBoxLayout()
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("/path/to/output_root")
        row.addWidget(self._root_edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_root)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Positions ────────────────────────────────────────────────────
        layout.addWidget(QLabel("Positions (comma-separated, e.g. 0,1,2)"))
        self._positions_edit = QLineEdit("0")
        layout.addWidget(self._positions_edit)

        # ── Timepoints ───────────────────────────────────────────────────
        self._tp_all_check = QCheckBox("All timepoints")
        self._tp_all_check.setChecked(True)
        self._tp_all_check.toggled.connect(self._on_tp_toggle)
        layout.addWidget(self._tp_all_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Timepoints (comma-separated)"))
        self._tp_edit = QLineEdit()
        self._tp_edit.setPlaceholderText("0,1,2,3,4")
        self._tp_edit.setEnabled(False)
        row.addWidget(self._tp_edit)
        layout.addLayout(row)

        # ── XY downsample ────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("XY downsample factor"))
        self._xy_spin = QSpinBox()
        self._xy_spin.setRange(1, 16)
        self._xy_spin.setValue(3)
        row.addWidget(self._xy_spin)
        layout.addLayout(row)

        # ── Overwrite ────────────────────────────────────────────────────
        self._overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self._overwrite_check)

        # ── Run button ───────────────────────────────────────────────────
        self._run_btn = QPushButton("Export Raw Data")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # ── Progress ─────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self.setLayout(layout)

    # ── Browsing ─────────────────────────────────────────────────────────

    def _browse_ndtiff(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select NDTiff directory")
        if d:
            self._ndtiff_edit.setText(d)

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output root directory")
        if d:
            self._root_edit.setText(d)

    def _on_tp_toggle(self, checked: bool) -> None:
        self._tp_edit.setEnabled(not checked)

    # ── Parsing ──────────────────────────────────────────────────────────

    def _parse_int_list(self, text: str) -> list[int]:
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    def _build_config(self) -> DatasetConfig:
        tp = None if self._tp_all_check.isChecked() else self._parse_int_list(self._tp_edit.text())
        return DatasetConfig(
            ndtiff_path=self._ndtiff_edit.text().strip(),
            root_dir=self._root_edit.text().strip(),
            positions=self._parse_int_list(self._positions_edit.text()),
            timepoints=tp,
            xy_downsample=self._xy_spin.value(),
        )

    # ── Run ──────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        try:
            config = self._build_config()
        except Exception as e:
            self._status_label.setText(f"Config error: {e}")
            return

        if not config.ndtiff_path or not config.root_dir:
            self._status_label.setText("Please set both NDTiff and output directories.")
            return
        if not config.positions:
            self._status_label.setText("Please specify at least one position.")
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Starting…")

        positions = list(config.positions)
        overwrite = self._overwrite_check.isChecked()

        self._n_positions = len(positions)

        @thread_worker(
            connect={
                "yielded": self._on_progress,
                "finished": self._on_finished,
                "errored": self._on_error,
            }
        )
        def _work():
            for pos in positions:
                for done, total, label in run_s00(config, pos, overwrite=overwrite):
                    yield (pos, done, total, label)

        self._worker = _work()

    def _on_progress(self, update: tuple) -> None:
        pos, done, total, label = update
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status_label.setText(f"pos{pos:02d} — {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Done — exported {self._n_positions} position(s).")
        self._worker = None

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None
