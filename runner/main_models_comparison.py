from pathlib import Path
import os

from momentum_health_walking_reconstruction import (
    KinematicsData,
    BiorbdKinematicsData,
    GlbAsCsvKinematicsData,
    Side,
    GaitCycle,
    SwayTrial,
    TrialType,
    AnalysesData,
    SwayDirection,
)


def _load_single_file(data_folder: Path, filter: str, expected_extension: str) -> Path:
    files = list(data_folder.glob(f"*{filter}*.{expected_extension}"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one {expected_extension.upper()} file for filter '{filter}' in folder '{data_folder}', but found {len(files)}."
        )
    return files[0]


def _get_kinematics_metrics(
    data_base_folder: Path,
    model_base_folder: Path,
    kinematics_base_folder: Path,
    subject: str,
    trial_type: TrialType,
    show_plot: bool = False,
) -> dict[str, dict[str, AnalysesData | list[AnalysesData]]]:
    if trial_type == TrialType.SWAY:
        glb_filter = "685"
        inhouse_filter = "P21_quiet_stand_01"
    elif trial_type == TrialType.GAIT:
        glb_filter = "700"
        inhouse_filter = "walk_5m"
    else:
        raise ValueError(f"Could not determine trial type from file path: {trial_type}")

    # Load the momentum_health data
    csv_file_path = _load_single_file(data_base_folder / "Fichiers_Momentum_Health" / subject, glb_filter, "csv")
    momentum_health = GlbAsCsvKinematicsData.from_file(csv_path=csv_file_path, trial_type=trial_type)

    # Load the in-house data
    model_path = f"{model_base_folder}/{subject}/lower_body.bioMod"
    inhouse_file_path = _load_single_file(kinematics_base_folder / subject, inhouse_filter, "npy")
    c3d_file_path = _load_single_file(data_base_folder / "inhouse_data" / subject, inhouse_filter, "c3d")
    inhouse = BiorbdKinematicsData.from_file(
        model_path=model_path, c3d_path=c3d_file_path, kinematics_path=inhouse_file_path, trial_type=trial_type
    )

    # Align the data together
    KinematicsData.perform_align_kinematics_data(inhouse, momentum_health, Side.LEFT, show_plot=True)

    metrics = {}
    if trial_type == TrialType.SWAY:
        metrics["sway"] = {}

        metrics["sway"]["inhouse"] = inhouse.extract_sway_trial(show_plot=show_plot)
        metrics["sway"]["momentum_health"] = momentum_health.extract_sway_trial(show_plot=show_plot)

    elif trial_type == TrialType.GAIT:
        metrics["left_cycles"] = {}
        metrics["right_cycles"] = {}

        metrics["left_cycles"]["inhouse"] = inhouse.extract_gait_cycles(side=Side.LEFT, show_plot=show_plot)
        metrics["left_cycles"]["momentum_health"] = momentum_health.extract_gait_cycles(
            side=Side.LEFT, show_plot=show_plot
        )

        metrics["right_cycles"]["inhouse"] = inhouse.extract_gait_cycles(side=Side.RIGHT, show_plot=show_plot)
        metrics["right_cycles"]["momentum_health"] = momentum_health.extract_gait_cycles(
            side=Side.RIGHT, show_plot=show_plot
        )
    else:
        raise ValueError(f"Unsupported trial type: {trial_type}")

    return metrics


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    model_base_folder = Path(os.getenv("MODELS_BASE_FOLDER"))
    kinematics_base_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    subject_names = os.getenv("SUBJECT_NAMES", "").split(",")

    trial_type_filter = os.getenv("TRIAL_FILE_NAME_FILTER")
    if "quiet_stand" in trial_type_filter.lower():
        trial_type = TrialType.SWAY
    elif "walk_5m" in trial_type_filter.lower():
        trial_type = TrialType.GAIT
    else:
        raise ValueError(f"Could not determine trial type from file path: {trial_type_filter}")

    metrics = {}
    for subject in subject_names:
        print(f"Processing subject {subject} for trial type {trial_type.value}...")
        metrics[subject] = _get_kinematics_metrics(
            data_base_folder=data_base_folder,
            model_base_folder=model_base_folder,
            kinematics_base_folder=kinematics_base_folder,
            subject=subject,
            trial_type=trial_type,
            show_plot=False,
        )

        # Show metrics
        if trial_type == TrialType.SWAY:
            inhouse_sway: SwayTrial = metrics[subject]["sway"]["inhouse"]
            momentum_health_sway: SwayTrial = metrics[subject]["sway"]["momentum_health"]

            print(
                f"Inhouse anteroposterior sway amplitude: {inhouse_sway.amplitude(direction=SwayDirection.ANTERO_POSTERIOR) * 1000:.2f} mm"
            )
            print(
                f"Momentum Health anteroposterior sway amplitude: {momentum_health_sway.amplitude(direction=SwayDirection.ANTERO_POSTERIOR) * 1000:.2f} mm"
            )

            print(
                f"Inhouse mediolateral sway amplitude: {inhouse_sway.amplitude(direction=SwayDirection.MEDIO_LATERAL) * 1000:.2f} mm"
            )
            print(
                f"Momentum Health mediolateral sway amplitude: {momentum_health_sway.amplitude(direction=SwayDirection.MEDIO_LATERAL) * 1000:.2f} mm"
            )

            print(
                f"Inhouse sway mean velocity: {inhouse_sway.mean_velocity(direction=SwayDirection.HORIZONTAL_PLANE) * 1000:.2f} mm/s"
            )
            print(
                f"Momentum Health sway mean velocity: {momentum_health_sway.mean_velocity(direction=SwayDirection.HORIZONTAL_PLANE) * 1000:.2f} mm/s"
            )

            print(f"Inhouse sway confidence ellipse: {inhouse_sway.confidence_ellipse_area(confidence_level=0.95)}")
            print(
                f"Momentum Health sway confidence ellipse: {momentum_health_sway.confidence_ellipse_area(confidence_level=0.95)}"
            )

            print(f"Inhouse sway length: {inhouse_sway.length(direction=SwayDirection.HORIZONTAL_PLANE) * 1000:.2f} mm")
            print(
                f"Momentum Health sway length: {momentum_health_sway.length(direction=SwayDirection.HORIZONTAL_PLANE) * 1000:.2f} mm"
            )

        elif trial_type == TrialType.GAIT:
            left_cycles: list[GaitCycle] = metrics[subject]["left_cycles"]["inhouse"]
            right_cycles: list[GaitCycle] = metrics[subject]["right_cycles"]["inhouse"]
            inhouse_cycles = left_cycles + right_cycles

            left_cycles: list[GaitCycle] = metrics[subject]["left_cycles"]["momentum_health"]
            right_cycles: list[GaitCycle] = metrics[subject]["right_cycles"]["momentum_health"]
            momentum_health_cycles = left_cycles + right_cycles

            if not inhouse_cycles or not momentum_health_cycles:
                print("No gait cycles found.")
            else:
                print(f"Inhouse cycles gait speed: {GaitCycle.mean_gait_speed_from_cycles(inhouse_cycles):.2f} m/s")
                print(
                    f"Momentum Health cycles gait speed: {GaitCycle.mean_gait_speed_from_cycles(momentum_health_cycles):.2f} m/s"
                )

                print(f"Inhouse cycles stride length: {GaitCycle.mean_stride_length_from_cycles(inhouse_cycles):.2f} m")
                print(
                    f"Momentum Health cycles stride length: {GaitCycle.mean_stride_length_from_cycles(momentum_health_cycles):.2f} m"
                )

                print(f"Inhouse cycles stride time: {GaitCycle.mean_stride_time_from_cycles(inhouse_cycles):.2f} s")
                print(
                    f"Momentum Health cycles stride time: {GaitCycle.mean_stride_time_from_cycles(momentum_health_cycles):.2f} s"
                )

        else:
            raise ValueError(f"Unsupported trial type: {trial_type}")


if __name__ == "__main__":
    main()
