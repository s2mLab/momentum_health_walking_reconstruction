import logging
import os
from pathlib import Path

import numpy as np

from ..kinematics.kinematics_reconstruction import kinematics_reconstruction, ReconstructionMethod
from ..models.visualizer import Visualizer


def reconstruct_all_kinematics(
    data_base_folder: Path,
    models_base_folder: Path,
    subject_names: list[str],
    results_folder: Path,
    output_model_name: str = "lower_body.bioMod",
    override_existing_trials: bool = False,
    animate_models: bool = False,
):
    _logger = logging.getLogger(__name__)

    for subject in subject_names:
        _logger.info(f"Reconstructing kinematics for subject {subject}...")

        # Prepare paths
        data_folder = data_base_folder / subject
        model_path = models_base_folder / subject / output_model_name
        result_folder = results_folder / subject
        trial_files = data_folder.glob("*.c3d")

        visualizer = Visualizer(model_path=model_path) if animate_models else None

        for trial in trial_files:
            trial_name = trial.stem
            _logger.info(f"  Processing: {trial_name}")

            output_filepath = result_folder / f"{trial_name}_q.npy"

            # Reconstruct kinematics
            if not override_existing_trials and output_filepath.exists():
                _logger.info(f"  Result file already exists and override is set to False, skipping.")
                continue
            q = kinematics_reconstruction(
                data_path=trial,
                model_path=model_path,
                visualizer=visualizer,
                reconstruction_method=ReconstructionMethod.KALMAN,
            )

            # Save results
            os.makedirs(output_filepath.parent, exist_ok=True)

            # Save the kinematics data
            np.save(output_filepath.as_posix(), q)
