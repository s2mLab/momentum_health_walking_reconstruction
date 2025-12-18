from enum import Enum
from pathlib import Path

import biorbd
import numpy as np
from scipy import optimize

from ..utils.data_markers import DataMarkers
from ..models.visualizer import Visualizer


class ReconstructionMethod(Enum):
    QLD = 1
    KALMAN = 2


def _qld_inverse_kinematics(model: biorbd.Biorbd, data: DataMarkers, visualizer: Visualizer | None) -> np.ndarray:
    """Perform inverse kinematics using the QLD algorithm.

    Parameters
    ----------
    model : biorbd.Biorbd
        The biomechanical model.
    data : DataMarkers
        The marker data.
    visualizer : Visualizer, optional
        The visualizer to update during reconstruction.

    Returns
    -------
    np.ndarray
        The estimated generalized coordinates.
    """

    def objective_function(q: np.ndarray, model: biorbd.Biorbd, target_markers: np.ndarray) -> list[np.ndarray]:
        residual = [
            (marker.world - target_markers[:3, i]) for i, marker in enumerate(model.markers(q)) if marker.is_technical
        ]

        residual_vector = np.array(residual).reshape(-1)
        return residual_vector[~np.isnan(residual_vector)]

    results = []
    q_init = np.zeros(model.nb_q)
    for frame in data.to_biorbd():
        results.append(
            optimize.least_squares(
                objective_function,
                q_init,
                args=(model, frame),
                method="lm",
                xtol=1e-6,
                ftol=1e-6,
                gtol=1e-6,
                max_nfev=1000,
            ).x
        )
        q_init = results[-1]

        if visualizer is not None:
            visualizer.update_frame(q=q_init, markers=frame)

    return np.array(results).T


def _kalman_inverse_kinematics(model: biorbd.Biorbd, data: DataMarkers, visualizer: Visualizer | None) -> np.ndarray:
    """Perform inverse kinematics using an Extended Kalman Filter.

    Parameters
    ----------
    model : biorbd.Biorbd
        The biomechanical model.
    data : DataMarkers
        The marker data.
    visualizer : Visualizer, optional
        The visualizer to update during reconstruction.

    Returns
    -------
    np.ndarray
        The estimated generalized coordinates.
    """
    # Find a decent first frame
    q_init = _qld_inverse_kinematics(
        model=model, data=DataMarkers(data.marker_names, data.to_numpy()[:, :, 0]), visualizer=None
    )[:, 0]

    technical_indices = [i for i, marker in enumerate(model.markers) if marker.is_technical]
    reduced_marker_names = [model.markers[i].name for i in technical_indices]
    data_technical_markers = DataMarkers(reduced_marker_names, data.to_numpy()[:, technical_indices, :])

    frame_count = len(data_technical_markers)
    kalman = biorbd.ExtendedKalmanFilterMarkers(model, frequency=100, q_init=q_init)
    q_recons = np.ndarray((model.nb_q, frame_count))
    all_markers = data_technical_markers.to_biorbd()
    for i, (q_i, _, _) in enumerate(kalman.reconstruct_frames(all_markers=all_markers)):
        q_recons[:, i] = q_i

        if visualizer is not None:
            visualizer.update_frame(q=q_recons[:, i], markers=all_markers[i])

    return q_recons


def kinematics_reconstruction(
    data_path: Path,
    model_path: Path,
    reconstruction_method: ReconstructionMethod = ReconstructionMethod.KALMAN,
    visualizer: Visualizer = None,
) -> np.ndarray:
    model = biorbd.Biorbd(model_path.as_posix())

    # Load markers from c3d file
    data = DataMarkers.from_c3d(data_path).filter(expected_marker_names=[marker.name for marker in model.markers])

    if reconstruction_method == ReconstructionMethod.QLD:
        q_recons = _qld_inverse_kinematics(model=model, data=data, visualizer=visualizer)
    elif reconstruction_method == ReconstructionMethod.KALMAN:
        q_recons = _kalman_inverse_kinematics(model=model, data=data, visualizer=visualizer)
    else:
        raise ValueError(f"Reconstruction method {reconstruction_method} not recognized.")

    return q_recons
