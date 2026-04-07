import json
from pathlib import Path
import os

from momentum_health_walking_reconstruction import TrialType, Metrics, AnalysesData, KinematicsData, Side


page_break = '\n<div style="page-break-after: always;"></div>\n\n'


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    model_base_folder = Path(os.getenv("MODELS_BASE_FOLDER"))
    kinematics_base_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    subject_names = os.getenv("SUBJECT_NAMES", "").split(",")
    data_matching = json.load(open(os.getenv("DATA_MATCHING_JSON")))
    scalar_stats_to_compare = os.getenv("SCALAR_STATS_TO_COMPARE", "").split(",")
    time_series_list_stats_to_compare_tp = os.getenv("TIME_SERIES_LIST_STATS_TO_COMPARE")
    save_folder = Path(os.getenv("SAVE_FOLDER"))

    if len(scalar_stats_to_compare) < 2:
        raise ValueError(
            "At least two data types must be specified in the 'SCALAR_STATS_TO_COMPARE' environment variable, separated by commas."
        )

    if time_series_list_stats_to_compare_tp is None:
        raise ValueError(
            "At least two data types must be specified in the 'TIME_SERIES_LIST_STATS_TO_COMPARE' environment variable, separated by commas."
        )

    time_series_list_stats_to_compare = []
    for comparison in time_series_list_stats_to_compare_tp.split(";"):
        data_types = comparison.strip("[]").split(",")
        if len(data_types) != 2:
            raise ValueError(
                f"Each comparison in 'TIME_SERIES_LIST_STATS_TO_COMPARE' must contain exactly two data types, separated by a comma. Invalid comparison: '{comparison}'"
            )
        time_series_list_stats_to_compare.append(data_types)

    if save_folder is None:
        raise ValueError("Environment variable 'SAVE_FOLDER' is not set.")

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

    save_folder = save_folder / trial_type.name.lower()
    save_folder.mkdir(parents=True, exist_ok=True)

    data, _, data_types = Metrics.get_aligned_data(
        data_base_folder=data_base_folder,
        model_base_folder=model_base_folder,
        kinematics_base_folder=kinematics_base_folder,
        subjects=subject_names,
        trial_type=trial_type,
        trial_names=trial_names,
        data_matching=data_matching["data"],
        show_plot=False,
    )

    if trial_type == TrialType.SWAY:
        sides = [Side.NOT_SIDED]
    elif trial_type == TrialType.GAIT:
        sides = [Side.LEFT, Side.RIGHT]
    else:
        raise ValueError(f"Unsupported trial type: {trial_type}")

    indices = {}
    for subject_name, trial_data in data.items():
        indices[subject_name] = {}

        for trial_name, typed_data in trial_data.items():
            indices[subject_name][trial_name] = {}

            for side in sides:
                indices[subject_name][trial_name][side.name] = []

                reference_data: KinematicsData = typed_data["Inhouse"]
                if trial_type == TrialType.SWAY:
                    subtrials: list[AnalysesData] = [reference_data.extract_sway_trial()]
                elif trial_type == TrialType.GAIT:
                    subtrials: list[AnalysesData] = reference_data.extract_gait_cycles(side=side)
                else:
                    raise ValueError(f"Unsupported trial type: {trial_type}")

                for subtrial in subtrials:
                    resampled_indices = [int(idx) for idx in subtrial.indices()]
                    indices[subject_name][trial_name][side.name].append({})

                    for typed_name, kinematics_data in typed_data.items():
                        kinematics_data: KinematicsData
                        resample_ratio = kinematics_data.resample_ratio(resampled=True)
                        base_frame = kinematics_data.initial_frame_index(resampled=False)
                        real_indices = [int(idx * resample_ratio + base_frame) for idx in resampled_indices]

                        indices[subject_name][trial_name][side.name][-1][typed_name] = real_indices

    json.dump(indices, open(save_folder / "common_indices.json", "w"), indent=4)


if __name__ == "__main__":
    main()
