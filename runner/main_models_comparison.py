from momentum_health_walking_reconstruction import (
    BiorbdKinematicsModel,
    GlbKinematicsModel,
    Side,
    perform_align_kinematics,
)
from matplotlib import pyplot as plt


def main():
    momentum = GlbKinematicsModel.from_file(glb_path="results/P21_copy/fda_700_gait.glb")
    inhouse = BiorbdKinematicsModel.from_file(
        model_path="results/P21_copy/lower_body.bioMod",
        c3d_path="/home/pariterre/Documents/ShareFolder/Felipe/backup_c3d/P21_copy/P21_walk_5m.c3d",
        kinematics_path="results/P21_copy/P21_walk_5m_q.npy",
    )
    perform_align_kinematics(inhouse, momentum, Side.LEFT)

    # Compute cycles metrics
    left_cycles = inhouse.gait_cycles(Side.LEFT)
    right_cycles = inhouse.gait_cycles(Side.RIGHT)

    plt.figure()
    plt.plot(
        inhouse.time_vector(resampled=True),
        inhouse.knee_sagittal_angles(Side.LEFT, resampled=True),
        label="In-house (resampled)",
    )
    plt.plot(
        inhouse.time_vector(resampled=False),
        inhouse.knee_sagittal_angles(Side.LEFT, resampled=False),
        label="In-house (original)",
    )
    plt.plot(momentum.time_vector(), momentum.knee_sagittal_angles(Side.LEFT), label="Momentum")
    plt.legend()
    plt.title("Knee trajectory")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.show()


if __name__ == "__main__":
    main()
