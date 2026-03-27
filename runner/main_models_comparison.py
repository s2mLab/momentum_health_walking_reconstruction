from pathlib import Path
import os

from momentum_health_walking_reconstruction import (
    KinematicsData,
    BiorbdKinematicsData,
    GlbKinematicsData,
    Side,
    GaitCycle,
    SwayTrial,
    TrialType,
    AnalysesData,
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
) -> dict[str, dict[str, AnalysesData]]:
    if trial_type == TrialType.SWAY:
        glb_filter = "quiet_stand"
        inhouse_filter = "quiet_stand"
    elif trial_type == TrialType.GAIT:
        glb_filter = "gait"
        inhouse_filter = "walk_5m"
    else:
        raise ValueError(f"Could not determine trial type from file path: {trial_type}")

    data_folder = kinematics_base_folder / subject

    # Load the momentum_health data
    glb_file_path = _load_single_file(data_folder, glb_filter, "glb")
    momentum = GlbKinematicsData.from_file(glb_path=glb_file_path, trial_type=trial_type)

    # Load the in-house data
    model_path = f"{model_base_folder}/{subject}/lower_body.bioMod"
    inhouse_file_path = _load_single_file(data_folder, inhouse_filter, "npy")
    c3d_file_path = _load_single_file(data_base_folder / subject, inhouse_filter, "c3d")
    inhouse = BiorbdKinematicsData.from_file(
        model_path=model_path, c3d_path=c3d_file_path, kinematics_path=inhouse_file_path, trial_type=trial_type
    )

    # Align the data together
    KinematicsData.perform_align_kinematics_data(inhouse, momentum, Side.LEFT, show_plot=show_plot)

    metrics = {}
    if trial_type == TrialType.SWAY:
        metrics["sway"] = {}

        metrics["sway"]["inhouse"] = SwayTrial.extract(kinematics_data=inhouse, show_plot=show_plot)
        metrics["sway"]["momentum"] = SwayTrial.extract(kinematics_data=momentum, show_plot=show_plot)

    elif trial_type == TrialType.GAIT:
        metrics["left_cycles"] = {}
        metrics["right_cycles"] = {}

        metrics["left_cycles"]["inhouse"] = GaitCycle.extract_all(
            kinematics_data=inhouse, side=Side.LEFT, show_plot=show_plot
        )
        # metrics["left_cycles"]["momentum"] = GaitCycle.extract_all(
        #     kinematics_data=momentum, side=Side.LEFT, show_plot=show_plot
        # )

        metrics["right_cycles"]["inhouse"] = GaitCycle.extract_all(
            kinematics_data=inhouse, side=Side.RIGHT, show_plot=show_plot
        )
        # metrics["right_cycles"]["momentum"] = GaitCycle.extract_all(
        #     kinematics_data=momentum, side=Side.RIGHT, show_plot=show_plot
        # )
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

    for subject in subject_names:
        print(f"Processing subject {subject} for trial type {trial_type.value}...")
        metrics = _get_kinematics_metrics(
            data_base_folder=data_base_folder,
            model_base_folder=model_base_folder,
            kinematics_base_folder=kinematics_base_folder,
            subject=subject,
            trial_type=trial_type,
            show_plot=True,
        )

        # Show metrics
        if trial_type == TrialType.SWAY:
            inhouse = metrics["sway"]["inhouse"]
            sway = SwayTrial.extract(kinematics_data=inhouse, show_plot=False)
            print(f"Sway amplitude: {sway.amplitude(exclude_vertical=True).mean() * 1000:.2f} mm")
            print(f"Sway mean velocity: {sway.velocity(exclude_vertical=True).mean() * 1000:.2f} mm/s")
            print(f"Sway confidence ellipse: {sway.confidence_ellipse(confidence_level=0.95)}")
            print(f"Sway length: {sway.length(exclude_vertical=True) * 1000:.2f} mm")
        elif trial_type == TrialType.GAIT:
            left_cycles = metrics["left_cycles"]["inhouse"]
            if not left_cycles:
                print("No left gait cycles found.")
            else:
                first_cycle = left_cycles[0]
                print(f"First left cycle gait speed: {first_cycle.mean_gait_speed(exclude_vertical=True)} m/s")
                print(f"First left cycle stride length: {first_cycle.stride_length(exclude_vertical=True):.2f} m")
                print(f"First left cycle stance time: {first_cycle.stance_time:.2f} s")
                print(f"First left cycle swing time: {first_cycle.swing_time:.2f} s")
                print(f"First left cycle stride time: {first_cycle.stride_time:.2f} s")

            right_cycles = metrics["right_cycles"]["inhouse"]
            if not right_cycles:
                print("No right gait cycles found.")
            else:
                first_cycle = right_cycles[0]
                print(f"First right cycle gait speed: {first_cycle.mean_gait_speed(exclude_vertical=True)} m/s")
                print(f"First right cycle stride length: {first_cycle.stride_length(exclude_vertical=True):.2f} m")
                print(f"First right cycle stance time: {first_cycle.stance_time:.2f} s")
                print(f"First right cycle swing time: {first_cycle.swing_time:.2f} s")
                print(f"First right cycle stride time: {first_cycle.stride_time:.2f} s")

        else:
            raise ValueError(f"Unsupported trial type: {trial_type}")


if __name__ == "__main__":
    main()
