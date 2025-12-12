"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

from enum import Enum
import os
from pathlib import Path

import numpy as np
from biobuddy import (
    Axis,
    BiomechanicalModel,
    C3dData,
    Marker,
    Mesh,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    DeLevaTable,
    Sex,
    SegmentName,
    ViewAs,
    SegmentCoordinateSystemUtils,
    RotoTransMatrix,
    InertiaParameters,
    JointCenterTool,
    Score,
    MarkerWeight,
    Sara,
)


class Markers(Enum):
    LPSI = "LPSI"
    RPSI = "RPSI"
    LASI = "LASI"
    RASI = "RASI"

    LTHI = "LTHI"
    LTHIB = "LTHIB"
    LTHID = "LTHID"
    LKNEE = "LKNE"
    LKNEEM = "LKNEM"


class NexusMarkers:
    def __init__(self, file: C3dData):
        self._markers = {}
        for m in Markers:
            is_found = False
            for name in file.marker_names:
                if name.endswith(m.value):
                    self._markers[m] = name
                    is_found = True
                    break
            if not is_found:
                raise RuntimeError(f"Marker {m.value} not found in the C3D file.")

    def __getitem__(self, marker: Markers) -> str:
        return self._markers[marker]


def model_creation_from_measured_data(output_model_filepath: str, calibration_folder: Path, animate_model: bool = True):
    # Load all the required data files
    static_file = list(calibration_folder.glob("*static.c3d"))
    if len(static_file) != 1:
        raise RuntimeError(f"Expected exactly one static file in {calibration_folder}, found {len(static_file)}.")
    static_trial = C3dData(static_file[0].as_posix())

    left_hip_functionnal_file = list(calibration_folder.glob("*func_lhip.c3d"))
    if len(left_hip_functionnal_file) != 1:
        raise RuntimeError(
            f"Expected exactly one left hip functional file in {calibration_folder}, found {len(left_hip_functionnal_file)}."
        )
    left_hip_functionnal_trial = C3dData(left_hip_functionnal_file[0].as_posix())

    # Find the full name of each marker
    markers = NexusMarkers(static_trial)

    # Hip
    lpsi = markers[Markers.LPSI]
    rpsi = markers[Markers.RPSI]
    lasi = markers[Markers.LASI]
    rasi = markers[Markers.RASI]

    # LThigh
    lthi = markers[Markers.LTHI]
    lthib = markers[Markers.LTHIB]
    lthid = markers[Markers.LTHID]
    lknee = markers[Markers.LKNEE]
    lkneem = markers[Markers.LKNEEM]

    # Generate the personalized kinematic model
    model = BiomechanicalModel()

    model.add_segment(
        Segment(
            name="Pelvis",
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers([lpsi, rpsi, lasi, rasi]),
                first_axis=Axis(
                    name=Axis.Name.X,
                    start=SegmentCoordinateSystemUtils.mean_markers([lpsi, lasi]),
                    end=SegmentCoordinateSystemUtils.mean_markers([rpsi, rasi]),
                ),
                second_axis=Axis(
                    name=Axis.Name.Z,
                    start=SegmentCoordinateSystemUtils.mean_markers([lpsi, rpsi]),
                    end=SegmentCoordinateSystemUtils.mean_markers([lasi, rasi]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((lpsi, rpsi, rasi, lasi, lpsi), is_local=False),
        )
    )
    model.segments["Pelvis"].add_marker(Marker(lpsi, is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker(rpsi, is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker(lasi, is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker(rasi, is_technical=True, is_anatomical=True))

    mid_knee = SegmentCoordinateSystemUtils.mean_markers([lknee, lkneem])
    model.add_segment(
        Segment(
            name="LThigh",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lasi,
                first_axis=Axis(name=Axis.Name.Z, start=mid_knee, end=lasi),
                second_axis=Axis(name=Axis.Name.X, start=lknee, end=lkneem),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (lasi, lthi, lthib, lthid, lthi, lthid, mid_knee, lknee, lkneem),
                is_local=False,
            ),
        )
    )
    model.segments["LThigh"].add_marker(Marker(lthi, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthib, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthid, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lknee, is_technical=True, is_anatomical=True))
    model.segments["LThigh"].add_marker(Marker(lkneem, is_technical=True, is_anatomical=True))

    # reduced_model.add_segment(
    #     Segment(
    #         name="RTibia",
    #         parent_name="RFemur",
    #         rotations=Rotations.X,
    #         inertia_parameters=de_leva[SegmentName.SHANK],
    #         segment_coordinate_system=SegmentCoordinateSystem(
    #             origin=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
    #             first_axis=Axis(name=Axis.Name.X, start="RSPH", end="RLM"),
    #             second_axis=Axis(
    #                 name=Axis.Name.Z,
    #                 start=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]),
    #                 end=SegmentCoordinateSystemUtils.mean_markers(["RMFE", "RLFE"]),
    #             ),
    #             axis_to_keep=Axis.Name.Z,
    #         ),
    #         mesh=Mesh(("RMFE", "RSPH", "RLM", "RLFE"), is_local=False),
    #     )
    # )
    # reduced_model.segments["RTibia"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
    # reduced_model.segments["RTibia"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))

    # # The foot is a special case since the position of the ankle relatively to the foot length is not given in De Leva
    # # So here we assume that the foot com is in the middle of the three foot markers
    # foot_inertia_parameters = de_leva[SegmentName.FOOT]
    # rt_matrix = RotoTransMatrix()
    # rt_matrix.from_euler_angles_and_translation(
    #     angle_sequence="y",
    #     angles=np.array([-np.pi / 2]),
    #     translation=np.array([0.0, 0.0, 0.0]),
    # )
    # foot_inertia_parameters.center_of_mass = lambda m, bio: rt_matrix.rt_matrix @ np.nanmean(
    #     np.nanmean(np.array([m[name] for name in ["LSPH", "LLM", "LTT2"]]), axis=0)
    #     - np.nanmean(np.array([m[name] for name in ["LSPH", "LLM"]]), axis=0),
    #     axis=1,
    # )

    # reduced_model.add_segment(
    #     Segment(
    #         name="RFoot",
    #         parent_name="RTibia",
    #         rotations=Rotations.X,
    #         segment_coordinate_system=SegmentCoordinateSystem(
    #             origin=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]),
    #             first_axis=Axis(
    #                 Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers(["RSPH", "RLM"]), end="RTT2"
    #             ),
    #             second_axis=Axis(Axis.Name.X, start="RSPH", end="RLM"),
    #             axis_to_keep=Axis.Name.Z,
    #         ),
    #         inertia_parameters=foot_inertia_parameters,
    #         mesh=Mesh(("RLM", "RTT2", "RSPH", "RLM"), is_local=False),
    #     )
    # )
    # reduced_model.segments["RFoot"].add_marker(Marker("RTT2", is_technical=True, is_anatomical=True))

    # reduced_model.add_segment(
    #     Segment(
    #         name="LFemur",
    #         parent_name="Pelvis",
    #         rotations=Rotations.XY,
    #         inertia_parameters=de_leva[SegmentName.THIGH],
    #         segment_coordinate_system=SegmentCoordinateSystem(
    #             origin=lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
    #                 static_trial.values, None
    #             )
    #             - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
    #             first_axis=Axis(name=Axis.Name.X, start="LLFE", end="LMFE"),
    #             second_axis=Axis(
    #                 name=Axis.Name.Z,
    #                 start=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
    #                 end=SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"]),
    #             ),
    #             axis_to_keep=Axis.Name.Z,
    #         ),
    #         mesh=Mesh(
    #             (
    #                 lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
    #                     static_trial.values, None
    #                 )
    #                 - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
    #                 "LMFE",
    #                 "LLFE",
    #                 lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["LPSIS", "LASIS"])(
    #                     static_trial.values, None
    #                 )
    #                 - np.array([0.0, 0.0, 0.05 * total_height, 0.0]),
    #             ),
    #             is_local=False,
    #         ),
    #     )
    # )
    # reduced_model.segments["LFemur"].add_marker(Marker("LLFE", is_technical=True, is_anatomical=True))
    # reduced_model.segments["LFemur"].add_marker(Marker("LMFE", is_technical=True, is_anatomical=True))

    # reduced_model.add_segment(
    #     Segment(
    #         name="LTibia",
    #         parent_name="LFemur",
    #         rotations=Rotations.X,
    #         inertia_parameters=de_leva[SegmentName.SHANK],
    #         segment_coordinate_system=SegmentCoordinateSystem(
    #             origin=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
    #             first_axis=Axis(name=Axis.Name.X, start="LLM", end="LSPH"),
    #             second_axis=Axis(
    #                 name=Axis.Name.Z,
    #                 start=SegmentCoordinateSystemUtils.mean_markers(["LSPH", "LLM"]),
    #                 end=SegmentCoordinateSystemUtils.mean_markers(["LMFE", "LLFE"]),
    #             ),
    #             axis_to_keep=Axis.Name.Z,
    #         ),
    #         mesh=Mesh(("LMFE", "LSPH", "LLM", "LLFE"), is_local=False),
    #     )
    # )
    # reduced_model.segments["LTibia"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
    # reduced_model.segments["LTibia"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))

    # foot_inertia_parameters = de_leva[SegmentName.FOOT]
    # rt_matrix = RotoTransMatrix()
    # rt_matrix.from_euler_angles_and_translation(
    #     angle_sequence="y",
    #     angles=np.array([-np.pi / 2]),
    #     translation=np.array([0.0, 0.0, 0.0]),
    # )
    # foot_inertia_parameters.center_of_mass = lambda m, bio: rt_matrix.rt_matrix @ np.nanmean(
    #     np.nanmean(np.array([m[name] for name in ["LSPH", "LLM", "LTT2"]]), axis=0)
    #     - np.nanmean(np.array([m[name] for name in ["LSPH", "LLM"]]), axis=0),
    #     axis=1,
    # )

    # reduced_model.add_segment(
    #     Segment(
    #         name="LFoot",
    #         parent_name="LTibia",
    #         rotations=Rotations.X,
    #         segment_coordinate_system=SegmentCoordinateSystem(
    #             origin=SegmentCoordinateSystemUtils.mean_markers(["LSPH", "LLM"]),
    #             first_axis=Axis(
    #                 Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers(["LLM", "LSPH"]), end="LTT2"
    #             ),
    #             second_axis=Axis(Axis.Name.X, start="LLM", end="LSPH"),
    #             axis_to_keep=Axis.Name.Z,
    #         ),
    #         inertia_parameters=foot_inertia_parameters,
    #         mesh=Mesh(("LLM", "LTT2", "LSPH", "LLM"), is_local=False),
    #     )
    # )
    # reduced_model.segments["LFoot"].add_marker(Marker("LTT2", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = model.to_real(static_trial)

    # Set up the joint center identification tool
    joint_center_tool = JointCenterTool(model_real)
    # Example for the right hip
    joint_center_tool.add(
        Score(
            functional_c3d=left_hip_functionnal_trial,
            parent_name="Pelvis",
            child_name="LThigh",
            parent_marker_names=[lpsi, rpsi, lasi, rasi],
            child_marker_names=[lthi, lthib, lthid, lknee, lkneem],
        )
    )
    model_real = joint_center_tool.replace_joint_centers()

    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD_BIOVIZ, model_path=output_model_filepath)

    return model_real


def main():

    # Load the static trial
    output_model_filepath = f"lower_body.bioMod"

    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    calibration_folder = data_base_folder / "calibration_files"

    model_creation_from_measured_data(
        output_model_filepath=output_model_filepath, calibration_folder=calibration_folder, animate_model=True
    )


if __name__ == "__main__":
    main()
