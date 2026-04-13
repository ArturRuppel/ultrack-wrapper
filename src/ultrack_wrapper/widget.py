"""Main docked QTabWidget for the Ultrack Wrapper plugin."""

from __future__ import annotations

from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ultrack_wrapper._widget_data_prep import DataPrepWidget
from ultrack_wrapper.widgets import CellposeWidget, UltrackAnalysisWidget


class UltrackWidget(QTabWidget):
    """Main plugin widget with one tab per pipeline stage."""

    def __init__(self, napari_viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = napari_viewer

        self.addTab(DataPrepWidget(napari_viewer), "Data Prep")
        self.addTab(CellposeWidget(napari_viewer), "Cellpose")
        self.addTab(UltrackAnalysisWidget(napari_viewer), "Analysis")
        self.addTab(self._make_placeholder("Post-processing"), "Post-proc")

    @staticmethod
    def _make_placeholder(name: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{name} — not yet implemented"))
        w.setLayout(layout)
        return w
