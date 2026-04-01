from pathlib import Path

import numpy as np

from .kinematics_data import (
    BiorbdKinematicsData,
    MomentumHealthCsvKinematicsData,
    PigKinematicsData,
    Joint,
    KinematicsData,
    TrialType,
    Side,
)
from ..utils.analyses_data import GaitCycle, GaitMetrics, SwayDirection, SwayMetrics, SwayTrial


class Metrics:
    @staticmethod
    def get_mean_metrics(
        data_base_folder: Path,
        model_base_folder: Path,
        kinematics_base_folder: Path,
        subjects: list[str],
        trial_type: TrialType,
        trial_names: list[str],
        data_matching: dict,
        show_plot: bool = False,
    ) -> dict:
        data_types, metrics = _get_trials_metrics(
            data_base_folder=data_base_folder,
            model_base_folder=model_base_folder,
            kinematics_base_folder=kinematics_base_folder,
            subjects=subjects,
            trial_type=trial_type,
            trial_names=trial_names,
            data_matching=data_matching,
            show_plot=show_plot,
        )

        # Show metrics
        if trial_type == TrialType.SWAY:
            metrics_to_fetch = [
                [
                    SwayMetrics.AMPLITUDE_AP,
                    SwayTrial.mean_amplitude_from_trials,
                    {"direction": SwayDirection.ANTERO_POSTERIOR},
                    lambda x: x * 1000,  # Convert from meters to millimeters
                ],
                [
                    SwayMetrics.AMPLITUDE_ML,
                    SwayTrial.mean_amplitude_from_trials,
                    {"direction": SwayDirection.MEDIO_LATERAL},
                    lambda x: x * 1000,  # Convert from meters to millimeters
                ],
                [
                    SwayMetrics.LENGTH,
                    SwayTrial.mean_length_from_trials,
                    {"direction": SwayDirection.HORIZONTAL_PLANE},
                    lambda x: x * 1000,  # Convert from meters to millimeters
                ],
                [
                    SwayMetrics.VELOCITY,
                    SwayTrial.mean_mean_velocity_from_trials,
                    {"direction": SwayDirection.HORIZONTAL_PLANE},
                    lambda x: x * 1000,  # Convert from meters/second to millimeters/second
                ],
                [
                    SwayMetrics.CONFIDENCE_ELLIPSE_AREA,
                    SwayTrial.mean_confidence_ellipse_area_from_trials,
                    {"confidence_level": 0.95},
                    lambda x: x * 100 * 100,  # Convert from square meters to square centimeters
                ],
            ]

            mean_metrics = {
                metric[0]: {
                    data_type: {
                        subject: metric[3](metric[1](metrics[subject]["sway"][data_type], **metric[2]))
                        for subject in metrics.keys()
                    }
                    for data_type in data_types
                }
                for metric in metrics_to_fetch
            }

        elif trial_type == TrialType.GAIT:
            metrics_to_fetch = [
                [GaitMetrics.GAIT_SPEED, GaitCycle.mean_gait_speed_from_cycles, {}],
                [GaitMetrics.STRIDE_LENGTH, GaitCycle.mean_stride_length_from_cycles, {}],
                [GaitMetrics.STRIDE_TIME, GaitCycle.mean_stride_time_from_cycles, {}],
            ]

            mean_metrics = {
                metric[0]: {
                    data_type: {
                        subject: np.mean(
                            [
                                metric[1](*metrics[subject][f"{side.name}_cycles"][data_type], **metric[2])
                                for side in [Side.LEFT, Side.RIGHT]
                            ]
                        )
                        for subject in metrics.keys()
                    }
                    for data_type in data_types
                }
                for metric in metrics_to_fetch
            }

        else:
            raise ValueError(f"Unsupported trial type: {trial_type}")

        return mean_metrics


def _get_trials_metrics(
    data_base_folder: Path,
    model_base_folder: Path,
    kinematics_base_folder: Path,
    subjects: list[str],
    trial_type: TrialType,
    trial_names: list[str],
    data_matching: dict,
    show_plot: bool = False,
) -> tuple[list[str], dict]:

    all_data = {}
    all_metrics = {}
    for subject in subjects:
        metrics = {}
        for trial_name in trial_names:
            momentum_health_filter_a = data_matching[subject][trial_name]["cameraA"]
            momentum_health_filter_b = data_matching[subject][trial_name]["cameraB"]
            inhouse_filter = data_matching[subject][trial_name]["c3d"]
            pig_filter = inhouse_filter
            if momentum_health_filter_a is None or momentum_health_filter_b is None or inhouse_filter is None:
                raise ValueError(
                    f"Expected 'cameraA', 'cameraB', and 'inhouse' filters to be defined for subject '{subject}' "
                    f"and trial '{trial_name}' in the data matching JSON."
                )

            # Load the momentum_health data
            all_data: dict[str, KinematicsData] = {
                "Momentum Health A": MomentumHealthCsvKinematicsData.from_file(
                    csv_path=_load_single_file(
                        data_base_folder / "momentum_health_data" / subject, momentum_health_filter_a, "csv"
                    ),
                    trial_type=trial_type,
                ),
                "Momentum Health B": MomentumHealthCsvKinematicsData.from_file(
                    csv_path=_load_single_file(
                        data_base_folder / "momentum_health_data" / subject, momentum_health_filter_b, "csv"
                    ),
                    trial_type=trial_type,
                ),
                "Inhouse": BiorbdKinematicsData.from_file(
                    model_path=f"{model_base_folder}/{subject}/lower_body.bioMod",
                    kinematics_path=_load_single_file(kinematics_base_folder / subject, inhouse_filter, "npy"),
                    c3d_path=_load_single_file(data_base_folder / "inhouse_data" / subject, inhouse_filter, "c3d"),
                    trial_type=trial_type,
                ),
                # "Plug-in Gait": PigKinematicsData.from_file(
                #     c3d_path=_load_single_file(data_base_folder / "pig_data" / subject, pig_filter, "c3d"),
                #     trial_type=trial_type,
                # ),
            }

            # Align the data together
            KinematicsData.perform_align_kinematics_data(all_data["Momentum Health A"], all_data["Momentum Health B"])
            KinematicsData.perform_align_kinematics_data(all_data["Inhouse"], all_data["Momentum Health A"])
            # all_data["Plug-in Gait"].duplicate_alignment_data_from(reference=all_data["Inhouse"])

            if show_plot:
                for i, data_type in enumerate(all_data.keys()):
                    all_data[data_type].plot(
                        joint=Joint.KNEE,
                        side=Side.LEFT,
                        title="Knee Angles",
                        label=data_type,
                        show_now=(i == len(all_data) - 1),
                    )

            if trial_type == TrialType.SWAY:
                if "sway" not in metrics:
                    metrics["sway"] = {data_type: [] for data_type in all_data.keys()}

                for data_type in all_data.keys():
                    metrics["sway"][data_type].append(all_data[data_type].extract_sway_trial(show_plot=show_plot))

            elif trial_type == TrialType.GAIT:
                for side in [Side.LEFT, Side.RIGHT]:
                    if f"{side}_cycles" not in metrics:
                        metrics[f"{side.name}_cycles"] = {data_type: [] for data_type in all_data.keys()}

                    for data_type in all_data.keys():
                        metrics[f"{side.name}_cycles"][data_type].append(
                            all_data[data_type].extract_gait_cycles(side=side, show_plot=show_plot)
                        )

            else:
                raise ValueError(f"Unsupported trial type: {trial_type}")

        all_metrics[subject] = metrics

    return all_data.keys(), all_metrics


def _load_single_file(data_folder: Path, filter: str, expected_extension: str) -> Path:
    files = list(data_folder.glob(f"*{filter}*.{expected_extension}"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one {expected_extension.upper()} file for filter '{filter}' in folder '{data_folder}', but found {len(files)}."
        )
    return files[0]
