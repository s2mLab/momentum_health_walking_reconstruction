from momentum_health_walking_reconstruction import (
    KinematicsData,
    BiorbdKinematicsData,
    GlbKinematicsData,
    Side,
    GaitCycle,
)
from matplotlib import pyplot as plt


def main():
    momentum = GlbKinematicsData.from_file(glb_path="results/P21_copy/fda_700_gait.glb")
    inhouse = BiorbdKinematicsData.from_file(
        model_path="results/P21_copy/lower_body.bioMod",
        c3d_path="/home/pariterre/Documents/ShareFolder/Felipe/backup_c3d/P21_copy/P21_walk_5m.c3d",
        kinematics_path="results/P21_copy/P21_walk_5m_q.npy",
    )
    KinematicsData.perform_align_kinematics_data(inhouse, momentum, Side.LEFT, show_plot=False)

    # Compute cycles metrics
    left_cycles = GaitCycle.gait_cycles(kinematics_data=inhouse, side=Side.LEFT, show_plot=True)
    right_cycles = GaitCycle.gait_cycles(kinematics_data=inhouse, side=Side.RIGHT, show_plot=True)


if __name__ == "__main__":
    main()
