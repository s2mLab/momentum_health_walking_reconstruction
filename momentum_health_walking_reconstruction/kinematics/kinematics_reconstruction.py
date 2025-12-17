from pathlib import Path

import biorbd
import numpy as np
import ezc3d


def kinematics_reconstruction(data_path: Path, model_path: Path) -> np.ndarray:
    model = biorbd.Biorbd(model_path.as_posix())

    # Load markers from c3d file
    c3d = ezc3d.c3d(data_path.as_posix())
    markers = c3d

    kalman = biorbd.ExtendedKalmanFilterMarkers(model, frequency=100)
    q_recons = np.ndarray(target_q.shape)
    for i, (q_i, _, _) in enumerate(kalman.reconstruct_frames(markers)):
        q_recons[:, i] = q_i
