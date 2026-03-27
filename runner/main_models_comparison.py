from momentum_health_walking_reconstruction import (
    KinematicsData,
    BiorbdKinematicsData,
    GlbKinematicsData,
    Side,
    GaitCycle,
    SwayTrial,
    TrialType,
)


def main():
    glb_file_path = "results/P21/fda_700_gait.glb"
    if "balance" in glb_file_path.lower():
        trial_type = TrialType.SWAY
    elif "gait" in glb_file_path.lower():
        trial_type = TrialType.GAIT
    else:
        raise ValueError(f"Could not determine trial type from file path: {glb_file_path}")

    inhouse_file_name = "P21_walk_5m"
    model_path = "results/P21/lower_body.bioMod"
    c3d_file_path = f"/home/pariterre/Documents/ShareFolder/Felipe/backup_c3d/P21/{inhouse_file_name}.c3d"
    kinematics_file_path = f"results/P21/{inhouse_file_name}_q.npy"

    momentum = GlbKinematicsData.from_file(glb_path=glb_file_path, trial_type=trial_type)
    inhouse = BiorbdKinematicsData.from_file(
        model_path=model_path, c3d_path=c3d_file_path, kinematics_path=kinematics_file_path, trial_type=trial_type
    )
    KinematicsData.perform_align_kinematics_data(inhouse, momentum, Side.LEFT, show_plot=True)

    # Compute metrics
    if trial_type == TrialType.SWAY:
        sway = SwayTrial.extract(kinematics_data=inhouse, show_plot=False)
        print(f"Sway amplitude: {sway.amplitude(exclude_vertical=True).mean() * 1000:.2f} mm")
        print(f"Sway mean velocity: {sway.velocity(exclude_vertical=True).mean() * 1000:.2f} mm/s")
        print(f"Sway confidence ellipse: {sway.confidence_ellipse(confidence_level=0.95)}")
        print(f"Sway length: {sway.length(exclude_vertical=True) * 1000:.2f} mm")
    elif trial_type == TrialType.GAIT:
        left_cycles = GaitCycle.extract_all(kinematics_data=inhouse, side=Side.LEFT, show_plot=True)
        if not left_cycles:
            print("No left gait cycles found.")
        else:
            first_cycle = left_cycles[0]
            print(f"First left cycle gait speed: {first_cycle.mean_gait_speed(exclude_vertical=True)} m/s")
            print(f"First left cycle stride length: {first_cycle.stride_length:.2f} m")
            print(f"First left cycle stance time: {first_cycle.stance_time:.2f} s")
            print(f"First left cycle swing time: {first_cycle.swing_time:.2f} s")
            print(f"First left cycle stride time: {first_cycle.stride_time:.2f} s")
        right_cycles = GaitCycle.extract_all(kinematics_data=inhouse, side=Side.RIGHT, show_plot=True)
        if not right_cycles:
            print("No right gait cycles found.")
        else:
            first_cycle = right_cycles[0]
            print(f"First right cycle gait speed: {first_cycle.mean_gait_speed(exclude_vertical=True)} m/s")
            print(f"First right cycle stride length: {first_cycle.stride_length:.2f} m")
            print(f"First right cycle stance time: {first_cycle.stance_time:.2f} s")
            print(f"First right cycle swing time: {first_cycle.swing_time:.2f} s")
            print(f"First right cycle stride time: {first_cycle.stride_time:.2f} s")

    else:
        raise ValueError(f"Unsupported trial type: {trial_type}")


if __name__ == "__main__":
    main()
