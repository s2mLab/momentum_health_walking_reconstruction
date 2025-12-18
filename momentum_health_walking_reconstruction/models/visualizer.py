from pathlib import Path

import numpy as np


class Visualizer:
    def __init__(self, model_path: Path):
        import bioviz

        self._viz = bioviz.Viz(model_path.as_posix())

        # Set the mapping between virtual and experimental markers
        all_marker_names = [n.to_string() for n in self._viz.model.markerNames()]
        technical_marker_names = [n.to_string() for n in self._viz.model.technicalMarkerNames()]
        self._viz.virtual_to_experimental_markers_indices = [
            all_marker_names.index(name) if name in all_marker_names else None for name in technical_marker_names
        ]

    def update_kinematics(self, q: np.ndarray, refresh_window: bool = True):
        self._viz.set_q(q, refresh_window=refresh_window)

    def update_experimental_markers(self, markers: np.ndarray, refresh_window: bool = True):
        from pyomeca import Markers

        self._viz.set_experimental_markers(Markers(markers[:, :, None]), refresh_window=refresh_window)

    def update_all(self, q: np.ndarray, markers: np.ndarray):
        self.update_kinematics(q, refresh_window=False)
        self.update_experimental_markers(markers, refresh_window=True)
