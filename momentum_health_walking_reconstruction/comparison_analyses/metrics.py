import logging
from pathlib import Path

import numpy as np

from .kinematics_data import (
    BiorbdKinematicsData,
    EmptyKinematicsData,
    MomentumHealthCsvKinematicsData,
    PigKinematicsData,
    Joint,
    KinematicsData,
    TrialType,
    Side,
)
from ..utils.analyses_data import GaitCycle, GaitMetrics, SwayDirection, SwayMetrics, SwayTrial

_logger = logging.getLogger(__name__)


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
            scalar_metrics_to_fetch = [
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

            mean_scalar_metrics = {
                metric[0]: {
                    data_type: {
                        subject: metric[3](metric[1](metrics[subject]["sway"][data_type], **metric[2]))
                        for subject in metrics.keys()
                    }
                    for data_type in data_types
                }
                for metric in scalar_metrics_to_fetch
            }

            inhouse_trial_indices = {
                subject: {
                    side: [data.indices() for data in metrics[subject][f"sway"]["Inhouse"] if data is not None]
                    for side in [Side.LEFT, Side.RIGHT]
                }
                for subject in metrics.keys()
            }
            joint_angles_to_fetch = [Joint.TRUNK]

        elif trial_type == TrialType.GAIT:
            scalar_metrics_to_fetch = [
                [GaitMetrics.GAIT_SPEED, GaitCycle.mean_gait_speed_from_cycles, {}],
                [GaitMetrics.STRIDE_LENGTH, GaitCycle.mean_stride_length_from_cycles, {}],
                [GaitMetrics.STRIDE_TIME, GaitCycle.mean_stride_time_from_cycles, {}],
            ]

            mean_scalar_metrics = {
                metric[0]: {
                    data_type: {
                        subject: np.mean(
                            [
                                metric[1](metrics[subject][f"{side.name}_cycles"][data_type], **metric[2])
                                for side in [Side.LEFT, Side.RIGHT]
                            ]
                        )
                        for subject in metrics.keys()
                    }
                    for data_type in data_types
                }
                for metric in scalar_metrics_to_fetch
            }

            inhouse_trial_indices = {
                subject: {
                    side: [data.indices() for data in metrics[subject][f"{side.name}_cycles"]["Inhouse"]]
                    for side in [Side.LEFT, Side.RIGHT]
                }
                for subject in metrics.keys()
            }
            joint_angles_to_fetch = [Joint.PELVIS, Joint.HIP, Joint.KNEE, Joint.ANKLE]

        else:
            raise ValueError(f"Unsupported trial type: {trial_type}")

        mean_angles_metrics = {
            joint: {
                side: {
                    data_type: {
                        subject: [
                            _cut_kinematics_data(
                                kinematics_data=metrics[subject]["trial"][data_type],
                                joint=joint,
                                side=side,
                                indices_list=inhouse_trial_indices[subject][side],
                                to_degrees=True,
                            )
                        ]
                        for subject in metrics.keys()
                    }
                    for data_type in data_types
                }
                for side in [Side.LEFT, Side.RIGHT]
            }
            for joint in joint_angles_to_fetch
        }

        return mean_scalar_metrics, mean_angles_metrics


def _cut_kinematics_data(
    kinematics_data: KinematicsData, joint: Joint, side: Side, indices_list: list[tuple[int, int]], to_degrees: bool
) -> np.ndarray:
    if isinstance(kinematics_data, EmptyKinematicsData):
        return np.ndarray((0, 1000))

    raw_data = kinematics_data.angles(joint=joint, side=side)
    if to_degrees:
        raw_data = np.degrees(raw_data)
    raw_time_vector = kinematics_data.time_vector()

    cut_data = []
    new_frame_count = 1000
    for start_index, end_index in indices_list:
        if end_index > len(raw_time_vector):
            _logger.warning(
                f"End index {end_index} is out of bounds for the time vector of length {len(raw_time_vector)}. Skipping this trial segment."
            )
            continue

        cut_data.append(
            np.interp(
                np.linspace(raw_time_vector[start_index], raw_time_vector[end_index - 1], num=new_frame_count),
                raw_time_vector[start_index:end_index],
                raw_data[start_index:end_index],
            )
        )
    return np.array(cut_data)


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
            }
            all_data["Plug-in Gait"] = PigKinematicsData.from_file(
                c3d_path=_load_single_file(data_base_folder / "pig_data" / subject, pig_filter, "c3d"),
                min_last_frame_index=all_data["Inhouse"].original_last_frame_index(resampled=False),
                trial_type=trial_type,
            )

            # Align the data together
            KinematicsData.perform_align_kinematics_data(
                all_data["Momentum Health A"], all_data["Momentum Health B"], show_plot=show_plot
            )
            reference = (
                all_data["Momentum Health B"]
                if isinstance(all_data["Momentum Health A"], EmptyKinematicsData)
                else all_data["Momentum Health A"]
            )
            KinematicsData.perform_align_kinematics_data(all_data["Inhouse"], reference, show_plot=show_plot)
            all_data["Plug-in Gait"].duplicate_alignment_data_from(reference=all_data["Inhouse"])

            metrics["trial"] = all_data
            if show_plot:
                if trial_type == TrialType.SWAY:
                    pass
                elif trial_type == TrialType.GAIT:
                    for i, data_type in enumerate(all_data.keys()):
                        all_data[data_type].plot(
                            joint=Joint.KNEE,
                            side=Side.LEFT,
                            title="Knee Angles",
                            label=data_type,
                            show_now=(i == len(all_data) - 1),
                        )
                else:
                    raise ValueError(f"Unsupported trial type: {trial_type}")

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
                        metrics[f"{side.name}_cycles"][data_type].extend(
                            all_data[data_type].extract_gait_cycles(side=side, show_plot=show_plot)
                        )

            else:
                raise ValueError(f"Unsupported trial type: {trial_type}")

        all_metrics[subject] = metrics

    return all_data.keys(), all_metrics


def _load_single_file(data_folder: Path, filter: str, expected_extension: str) -> Path:
    if filter == "NONE":
        return None

    files = list(data_folder.glob(f"*{filter}*.{expected_extension}"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one {expected_extension.upper()} file for filter '{filter}' in folder '{data_folder}', but found {len(files)}."
        )
    return files[0]
