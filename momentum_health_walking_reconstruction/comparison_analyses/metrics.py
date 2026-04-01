from pathlib import Path

import numpy as np

from .kinematics_data import BiorbdKinematicsData, GlbAsCsvKinematicsData, Joint, KinematicsData, TrialType, Side
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
        metrics = _get_trials_metrics(
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
                    for data_type in ["inhouse", "momentum_health_a", "momentum_health_b"]
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
                    for data_type in ["inhouse", "momentum_health_a", "momentum_health_b"]
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
) -> dict:

    all_metrics = {}
    for subject in subjects:
        metrics = {}
        for trial_name in trial_names:
            inhouse_filter = data_matching[subject][trial_name]["c3d"]
            momentum_health_filter_a = data_matching[subject][trial_name]["cameraA"]
            momentum_health_filter_b = data_matching[subject][trial_name]["cameraB"]
            if inhouse_filter is None or momentum_health_filter_a is None or momentum_health_filter_b is None:
                raise ValueError(
                    f"Expected 'inhouse', 'cameraA' and 'cameraB' filters to be defined for subject '{subject}' and trial '{trial_name}' in the data matching JSON."
                )

            # Load the momentum_health data
            momentum_health_a_file_path = _load_single_file(
                data_base_folder / "momentum_health_data" / subject, momentum_health_filter_a, "csv"
            )
            momentum_health_b_file_path = _load_single_file(
                data_base_folder / "momentum_health_data" / subject, momentum_health_filter_b, "csv"
            )
            momentum_health_a = GlbAsCsvKinematicsData.from_file(
                csv_path=momentum_health_a_file_path, trial_type=trial_type
            )
            momentum_health_b = GlbAsCsvKinematicsData.from_file(
                csv_path=momentum_health_b_file_path, trial_type=trial_type
            )

            # Load the in-house data
            model_path = f"{model_base_folder}/{subject}/lower_body.bioMod"
            inhouse_file_path = _load_single_file(kinematics_base_folder / subject, inhouse_filter, "npy")
            c3d_file_path = _load_single_file(data_base_folder / "inhouse_data" / subject, inhouse_filter, "c3d")
            inhouse = BiorbdKinematicsData.from_file(
                model_path=model_path, c3d_path=c3d_file_path, kinematics_path=inhouse_file_path, trial_type=trial_type
            )

            # Align the data together
            KinematicsData.perform_align_kinematics_data(momentum_health_b, momentum_health_a, show_plot=False)
            KinematicsData.perform_align_kinematics_data(inhouse, momentum_health_a, show_plot=False)
            if show_plot:
                momentum_health_a.plot(
                    joint=Joint.KNEE, side=Side.LEFT, title="Knee Angles", label="Momentum Health A", show_now=False
                )
                momentum_health_b.plot(
                    joint=Joint.KNEE, side=Side.LEFT, title="Knee Angles", label="Momentum Health B", show_now=False
                )
                inhouse.plot(joint=Joint.KNEE, side=Side.LEFT, title="Knee Angles", label="Inhouse", show_now=True)

            if trial_type == TrialType.SWAY:
                if "sway" not in metrics:
                    metrics["sway"] = {"inhouse": [], "momentum_health_a": [], "momentum_health_b": []}

                metrics["sway"]["inhouse"].append(inhouse.extract_sway_trial(show_plot=show_plot))
                metrics["sway"]["momentum_health_a"].append(momentum_health_a.extract_sway_trial(show_plot=show_plot))
                metrics["sway"]["momentum_health_b"].append(momentum_health_b.extract_sway_trial(show_plot=show_plot))

            elif trial_type == TrialType.GAIT:
                for side in [Side.LEFT, Side.RIGHT]:
                    if f"{side}_cycles" not in metrics:
                        metrics[f"{side.name}_cycles"] = {
                            "inhouse": [],
                            "momentum_health_a": [],
                            "momentum_health_b": [],
                        }

                    metrics[f"{side.name}_cycles"]["inhouse"].append(
                        inhouse.extract_gait_cycles(side=side, show_plot=show_plot)
                    )
                    metrics[f"{side.name}_cycles"]["momentum_health_a"].append(
                        momentum_health_a.extract_gait_cycles(side=side, show_plot=show_plot)
                    )
                    metrics[f"{side.name}_cycles"]["momentum_health_b"].append(
                        momentum_health_b.extract_gait_cycles(side=side, show_plot=show_plot)
                    )

            else:
                raise ValueError(f"Unsupported trial type: {trial_type}")

        all_metrics[subject] = metrics

    return all_metrics


def _load_single_file(data_folder: Path, filter: str, expected_extension: str) -> Path:
    files = list(data_folder.glob(f"*{filter}*.{expected_extension}"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one {expected_extension.upper()} file for filter '{filter}' in folder '{data_folder}', but found {len(files)}."
        )
    return files[0]
