import logging
from pathlib import Path

from ..kinematics.kinematics_reconstruction import kinematics_reconstruction, ReconstructionMethod
from ..models.visualizer import Visualizer
from ..utils.data_markers import DataMarkers


def visualize_all_kinematics(
    data_base_folder: Path,
    models_base_folder: Path,
    subject_names: list[str],
    results_folder: Path,
    model_name: str = "lower_body.bioMod",
):
    _logger = logging.getLogger(__name__)

    visualizer = None
    for subject in subject_names:
        _logger.info(f"Visualizing kinematics for subject {subject}...")

        # Prepare paths
        data_folder = data_base_folder / subject
        model_path = models_base_folder / subject / model_name
        result_folder = results_folder / subject
        trial_files = data_folder.glob("*.c3d")

        if visualizer is None:
            visualizer = Visualizer(model_path=model_path)
        else:
            visualizer.swap_model(model_path=model_path)

        for trial in trial_files:
            trial_name = trial.stem
            _logger.info(f"  Processing: {trial_name}")

            kinematics_filepath = result_folder / f"{trial_name}_q.npy"

            # Load data
            if not kinematics_filepath.exists():
                _logger.info(f"  Result file not reconstructed, skipping.")
                continue
            visualizer.load_movement(kinematics_path=kinematics_filepath, markers_path=trial)

            # Wait until the user press "Enter" in the console to go to the next trial
            input("Press Enter to continue to the next trial...")
