import json
from pathlib import Path
import os

from momentum_health_walking_reconstruction import TrialType, Metrics
import numpy as np


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    model_base_folder = Path(os.getenv("MODELS_BASE_FOLDER"))
    kinematics_base_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    subject_names = os.getenv("SUBJECT_NAMES", "").split(",")
    data_matching = json.load(open(os.getenv("DATA_MATCHING_JSON")))

    trial_category = os.getenv("TRIAL_CATEGORY")
    if trial_category is None:
        raise ValueError(f"Environment variable 'TRIAL_CATEGORY' is not set.")

    trial_type = data_matching["trial_types"][trial_category]["category"]
    if trial_type == "gait":
        trial_type = TrialType.GAIT
    elif trial_type == "sway":
        trial_type = TrialType.SWAY
    else:
        raise ValueError(f"Could not determine trial type from file path: {trial_category}")

    trial_names = data_matching["trial_types"][trial_category]["trial_names"]
    if not isinstance(trial_names, list):
        raise ValueError(
            f"Expected 'trial_names' to be a list in the data matching JSON for trial category '{trial_category}'."
        )

    metrics = Metrics.get_mean_metrics(
        data_base_folder=data_base_folder,
        model_base_folder=model_base_folder,
        kinematics_base_folder=kinematics_base_folder,
        subjects=subject_names,
        trial_type=trial_type,
        trial_names=trial_names,
        data_matching=data_matching["data"],
        show_plot=False,
    )

    # Show metrics
    for metric_type, data in metrics.items():
        print(f"{metric_type.name.replace('_', ' ').title()}")
        for data_type, subject_data in data.items():
            mean = np.mean(list(subject_data.values()))
            std = np.std(list(subject_data.values()))
            print(f"\t{data_type.replace('_', ' ').title()}: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()
