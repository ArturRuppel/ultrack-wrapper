"""Main docked QTabWidget for the Ultrack Wrapper plugin."""

from __future__ import annotations

from qtpy.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel

from ultrack_wrapper._widget_data_prep import DataPrepWidget


class UltrackWidget(QTabWidget):
    """Main plugin widget with one tab per pipeline stage."""

    def __init__(self, napari_viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = napari_viewer

        self.addTab(DataPrepWidget(napari_viewer), "Data Prep")
        self.addTab(self._make_placeholder("Cellpose"), "Cellpose")
        self.addTab(self._make_placeholder("Foreground"), "Foreground")
        self.addTab(self._make_placeholder("Tracking"), "Tracking")
        self.addTab(self._make_placeholder("Post-processing"), "Post-proc")

    @staticmethod
    def _make_placeholder(name: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{name} — not yet implemented"))
        w.setLayout(layout)
        return w
