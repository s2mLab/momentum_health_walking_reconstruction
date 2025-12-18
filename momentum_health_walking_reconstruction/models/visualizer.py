from pathlib import Path

import biorbd
import numpy as np
from pyomeca import Markers


class Visualizer:
    def __init__(self, model_path: Path):
        import bioviz

        self._viz = bioviz.Viz(model_path.as_posix())

        # Set the mapping between virtual and experimental markers
        model_marker_names = [n.to_string() for n in self._viz.model.markerNames()]
        technical_marker_names = [n.to_string() for n in self._viz.model.technicalMarkerNames()]
        self._viz.virtual_to_experimental_markers_indices = [
            model_marker_names.index(name) if name in model_marker_names else None for name in technical_marker_names
        ]

    def swap_model(self, model_path: Path):

        # This is possible only because they all use the same Biomechanical model structure
        self._viz.model = biorbd.Model(model_path.as_posix())

    def update_frame(self, q: np.ndarray, markers: np.ndarray):
        self._viz.set_q(q, refresh_window=False)
        self._viz.set_experimental_markers(Markers(markers[:, :, None]), refresh_window=True)

    def load_movement(self, kinematics_path: Path, markers_path: Path):
        # Get the model
        model = self._viz.model

        # Load experimental markers
        self._viz.experimental_markers = Markers.from_c3d(markers_path)[:, : model.nbMarkers(), :]
        if self._viz.experimental_markers.units == "mm":
            self._viz.experimental_markers = self._viz.experimental_markers * 0.001

        model_marker_names = [marker.name().to_string() for marker in model.markers()]
        exp_marker_names = [name.split(":")[1] for name in self._viz.experimental_markers.channel.data]
        self._viz.virtual_to_experimental_markers_indices = [
            model_marker_names.index(name) if name in model_marker_names else None for name in exp_marker_names
        ]
        self._viz.show_experimental_markers = True

        # Load the kinematics
        q = np.load(kinematics_path.as_posix())
        self._viz.load_movement(q, auto_start=False)

        # Set the first frame
        self._viz._set_movement_slider()
        self._viz.refresh_window()
