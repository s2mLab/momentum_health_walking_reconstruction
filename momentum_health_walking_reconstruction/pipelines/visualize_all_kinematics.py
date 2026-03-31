import logging
from pathlib import Path

from ..models.visualizer import Visualizer


def visualize_all_kinematics(
    data_base_folder: Path,
    models_base_folder: Path,
    subject_names: list[str],
    results_folder: Path,
    trial_file_name_filters: list[str] | None = None,
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
        trial_files = []
        if trial_file_name_filters is None:
            trial_files = data_folder.glob("*.c3d")
        else:
            for filter in trial_file_name_filters:
                trial_files.extend(data_folder.glob(f"*{filter}*.c3d"))

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
            if visualizer is None:
                visualizer = Visualizer(model_path=model_path)

            visualizer.load_movement(kinematics_path=kinematics_filepath, markers_path=trial)

            # Wait until the user press "Enter" in the console to go to the next trial
            input("Press Enter to continue to the next trial...")
            if not visualizer._viz.vtk_window.is_active:
                # If the window was closed, we set the visualizer to None so that it will be re-created for the next subject
                visualizer = None
