import json
import logging
from pathlib import Path
import os

import matplotlib.pyplot as plt
from momentum_health_walking_reconstruction import TrialType, Metrics
import numpy as np


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    model_base_folder = Path(os.getenv("MODELS_BASE_FOLDER"))
    kinematics_base_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    subject_names = os.getenv("SUBJECT_NAMES", "").split(",")
    data_matching = json.load(open(os.getenv("DATA_MATCHING_JSON")))

    trial_category = os.getenv("TRIAL_TYPE")
    if trial_category is None:
        raise ValueError(f"Environment variable 'TRIAL_TYPE' is not set.")

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

    metrics, angles = Metrics.get_mean_metrics(
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
        for data_type, data_by_subjects in data.items():
            mean = np.nanmean(list(data_by_subjects.values()))
            std = np.nanstd(list(data_by_subjects.values()))
            print(f"\t{data_type.replace('_', ' ').title()}: {mean:.2f} ± {std:.2f}")

    data_by_joints = angles
    for joint, data_by_sides in data_by_joints.items():
        for side_index, (side, data_by_types) in enumerate(data_by_sides.items()):
            if side_index == 0:
                # Make n-side rows on the figure if there are n sides to compare, and share the x-axis
                if len(data_by_sides) > 1:
                    fig, axes = plt.subplots(nrows=len(data_by_sides), ncols=1, sharex=True)
                else:
                    fig = plt.figure()
                title = f"{joint.name.replace('_', ' ').title()} angles comparison"
                fig.canvas.manager.set_window_title(title)
                fig.suptitle(title)
                fig.set_size_inches(10, 4 * len(data_by_sides))

            # Select the subplot according to the side index if there are multiple sides, otherwise use the single figure
            if len(data_by_sides) > 1:
                plt.sca(axes[side_index])

            for data_type, data_by_subjects in data_by_types.items():
                all_subjects_mean_data = []
                for subject, trials_data in data_by_subjects.items():
                    data = np.nanmean(np.array(trials_data), axis=1).mean(axis=0)
                    if not data.shape:
                        logging.warning(
                            f"Warning: Data for {data_type} could not be averaged across trials for subject {subject}. Skipping this subject."
                        )
                        continue
                    all_subjects_mean_data.append(data)
                all_subjects_mean_data = np.array(all_subjects_mean_data)
                mean = np.nanmean(all_subjects_mean_data, axis=0)
                std = np.nanstd(all_subjects_mean_data, axis=0)
                time = np.linspace(0, 100, mean.shape[0])
                plt.plot(time, mean, label=f"{data_type.replace('_', ' ').title()}")
                plt.fill_between(time, mean - std, mean + std, alpha=0.2)
            if len(data_by_sides) > 1:
                plt.title(f"{side.name.title()}")
            plt.xlabel("Gait cycle (%)")
            plt.ylabel("Angle (degrees)")
            if side_index == len(data_by_sides) - 1:
                plt.legend()

    plt.show()
    logging.info("Done.")


if __name__ == "__main__":
    main()
