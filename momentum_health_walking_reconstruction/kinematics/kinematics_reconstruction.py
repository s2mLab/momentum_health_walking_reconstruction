from enum import Enum
from pathlib import Path

import biorbd
import numpy as np
from scipy import optimize

from ..utils.data_markers import DataMarkers


class ReconstructionMethod(Enum):
    QLD = 1
    KALMAN = 2


def _qld_inverse_kinematics(model: biorbd.Biorbd, markers: DataMarkers) -> np.ndarray:
    """Perform inverse kinematics using the QLD algorithm.

    Parameters
    ----------
    model : biorbd.Biorbd
        The biomechanical model.
    markers : DataMarkers
        The marker data.

    Returns
    -------
    np.ndarray
        The estimated generalized coordinates.
    """

    def objective_function(q: np.ndarray, model: biorbd.Biorbd, target_markers: np.ndarray) -> list[np.ndarray]:
        residual = [(marker.world - target_markers[:3, i]) for i, marker in enumerate(model.markers(q))]
        return np.array(residual).reshape(-1)

    results = []
    q_init = np.zeros(model.nb_q)
    for frame in range(len(markers)):
        markers_frame = markers.to_numpy()[:, :, frame]
        if np.any(np.isnan(markers_frame)):
            results.append(results[-1] if results else q_init)
            continue

        results.append(
            optimize.least_squares(
                objective_function,
                q_init,
                args=(model, markers_frame),
                method="lm",
                xtol=1e-6,
                ftol=1e-6,
                gtol=1e-6,
                max_nfev=1000,
            ).x
        )
        q_init = results[-1]
    return np.array(results).T


def _kalman_inverse_kinematics(model: biorbd.Biorbd, markers: DataMarkers) -> np.ndarray:
    """Perform inverse kinematics using an Extended Kalman Filter.

    Parameters
    ----------
    model : biorbd.Biorbd
        The biomechanical model.
    markers : DataMarkers
        The marker data.

    Returns
    -------
    np.ndarray
        The estimated generalized coordinates.
    """
    # Find a decent first frame
    q_init = _qld_inverse_kinematics(
        model=model, markers=DataMarkers(markers.marker_names, markers.to_numpy()[:, :, 0])
    )[:, 0]

    frame_count = len(markers)
    kalman = biorbd.ExtendedKalmanFilterMarkers(model, frequency=100, q_init=q_init)
    q_recons = np.ndarray((model.nb_q, frame_count))
    for i, (q_i, _, _) in enumerate(kalman.reconstruct_frames(all_markers=markers.to_biorbd())):
        q_recons[:, i] = q_i
    return q_recons


def kinematics_reconstruction(
    data_path: Path,
    model_path: Path,
    reconstruction_method: ReconstructionMethod = ReconstructionMethod.KALMAN,
    show: bool = False,
) -> np.ndarray:
    model = biorbd.Biorbd(model_path.as_posix())

    # Load markers from c3d file
    data = DataMarkers.from_c3d(data_path).filter(expected_marker_names=[marker.name for marker in model.markers])

    if reconstruction_method == ReconstructionMethod.QLD:
        q_recons = _qld_inverse_kinematics(model=model, markers=data)
    elif reconstruction_method == ReconstructionMethod.KALMAN:
        q_recons = _kalman_inverse_kinematics(model=model, markers=data)
    else:
        raise ValueError(f"Reconstruction method {reconstruction_method} not recognized.")

    if show:
        import bioviz

        viz = bioviz.Viz(model_path.as_posix())
        viz.load_movement(q_recons)
        viz.load_experimental_markers(data.to_bioviz())
        viz.exec()
