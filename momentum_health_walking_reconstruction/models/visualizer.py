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
        self._viz.Markers.m = self._viz.model
        self._viz.CoM.m = self._viz.model
        self._viz.CoMbySegment.m = self._viz.model
        self._viz.meshPointsInMatrix.m = self._viz.model
        self._viz.allGlobalJCS.m = self._viz.model
        self._viz.set_q(np.zeros(self._viz.model.nbQ()), refresh_window=True)

    def update_frame(self, q: np.ndarray, markers: np.ndarray):
        self._viz.set_q(q, refresh_window=False)
        self._viz.set_experimental_markers(Markers(markers[:, :, None]), refresh_window=True)

    def load_movement(
        self, kinematics_path: Path = None, kinematics_array: np.ndarray = None, markers_path: Path = None
    ):
        # Get the model
        model = self._viz.model

        if markers_path is not None:
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
        if kinematics_path is not None and kinematics_array is not None:
            raise ValueError("Only one of kinematics_path or kinematics_array should be provided.")
        elif kinematics_array is not None:
            q = kinematics_array
            self._viz.load_movement(q, auto_start=False)
        elif kinematics_path is not None:
            q = np.load(kinematics_path.as_posix())
            self._viz.load_movement(q, auto_start=False)

        # Set the first frame
        self._viz.set_q(q[:, 0], refresh_window=True)
        self._viz._set_movement_slider()
        self._viz._animate_from_slider()
        self._viz.refresh_window()
