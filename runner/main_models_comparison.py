import json
import logging
from pathlib import Path
import os

from momentum_health_walking_reconstruction import TrialType, Metrics, Joint, GenerateReport, Io


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    model_base_folder = Path(os.getenv("MODELS_BASE_FOLDER"))
    kinematics_base_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    data_matching = json.load(open(os.getenv("DATA_MATCHING_JSON")))
    scalar_stats_to_compare = Io.parse_vector_env_variable("SCALAR_STATS_TO_COMPARE", minimum_length=2)
    time_series_list_stats_to_compare = Io.parse_multivectors_env_variable(
        "TIME_SERIES_LIST_STATS_TO_COMPARE", inner_minimum_length=2, outer_minimum_length=1
    )
    trial_categories = Io.parse_vector_env_variable("TRIAL_CATEGORIES", minimum_length=1)
    subject_names = Io.parse_multivectors_env_variable(
        "SUBJECT_NAMES",
        inner_minimum_length=1,
        outer_minimum_length=len(trial_categories),
        outer_maximum_length=len(trial_categories),
    )
    base_save_folder = Path(os.getenv("SAVE_FOLDER"))

    if base_save_folder is None:
        raise ValueError("Environment variable 'SAVE_FOLDER' is not set.")

    for trial_category, subjects in zip(trial_categories, subject_names):
        logging.info(f"Generating report for trial category: {trial_category}")

        # Prepare all the required data
        trial_type, data, metrics, data_types = Metrics.get_aligned_data(
            data_base_folder=data_base_folder,
            model_base_folder=model_base_folder,
            kinematics_base_folder=kinematics_base_folder,
            subjects=subjects,
            trial_category=trial_category,
            data_matching=data_matching,
            show_plot=False,
        )

        if trial_type == TrialType.GAIT:
            joint_angles_to_fetch = [Joint.PELVIS, Joint.HIP, Joint.KNEE, Joint.ANKLE]
        elif trial_type == TrialType.SWAY:
            joint_angles_to_fetch = [Joint.TRUNK]
        else:
            raise ValueError(f"Unsupported trial type: {trial_type}")
        scalar_metrics, angles = Metrics.get_mean_metrics(
            metrics=metrics, data_types=data_types, trial_type=trial_type, joint_angles_to_fetch=joint_angles_to_fetch
        )

        # Write the results files
        save_folder = base_save_folder / trial_category.lower()
        save_folder.mkdir(parents=True, exist_ok=True)
        GenerateReport.write_cutter_to_file(data=data, trial_type=trial_type, save_folder=save_folder)
        GenerateReport.write_metrics_to_file(
            scalar_metrics=scalar_metrics,
            angles=angles,
            trial_type=trial_type,
            save_folder=save_folder,
            scalar_stats_to_compare=scalar_stats_to_compare,
            time_series_list_stats_to_compare=time_series_list_stats_to_compare,
        )

    logging.info("Generating report done.")


if __name__ == "__main__":
    main()
