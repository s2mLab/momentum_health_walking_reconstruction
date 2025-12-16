"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

from enum import Enum
import os
from pathlib import Path

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
    ViewAs,
    SegmentCoordinateSystemUtils,
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

    LTIB = "LTIB"
    LTIBF = "LTIBF"
    LTIBD = "LTIBD"
    LANK = "LANK"
    LANKM = "LANKM"

    LHEE = "LHEE"
    LNAV = "LNAV"
    LTOE = "LTOE"
    LTOE5 = "LTOE5"

    RTHI = "RTHI"
    RTHIB = "RTHIB"
    RTHID = "RTHID"
    RKNEE = "RKNE"
    RKNEEM = "RKNEM"

    RTIB = "RTIB"
    RTIBF = "RTIBF"
    RTIBD = "RTIBD"
    RANK = "RANK"
    RANKM = "RANKM"

    RHEE = "RHEE"
    RNAV = "RNAV"
    RTOE = "RTOE"
    RTOE5 = "RTOE5"


class NexusMarkers:
    def __init__(self, file: C3dData):
        self._markers = {}
        for m in Markers:
            is_found = False
            for name in file.marker_names:
                name: str
                if name.endswith(m.value):
                    self._markers[m] = name
                    is_found = True
                    break
            if not is_found:
                raise RuntimeError(f"Marker {m.value} not found in the C3D file.")

    def __getitem__(self, marker: Markers) -> str:
        return self._markers[marker]


def model_creation_from_measured_data(
    output_model_filepath: str, calibration_folder: Path, animate_model: bool = True, use_score: bool = True
):
    # --- Load all the required data files --- #
    trials = {
        "static": "*static.c3d",
        "left_hip_functionnal": "*func_lhip.c3d",
        "left_knee_functionnal": "*func_lknee.c3d",
        "left_ankle_functionnal": "*func_lankle.c3d",
        "right_hip_functionnal": "*func_rhip.c3d",
        "right_knee_functionnal": "*func_rknee.c3d",
        "right_ankle_functionnal": "*func_rankle.c3d",
    }

    for key, pattern in trials.items():
        files = list(calibration_folder.glob(pattern))
        if len(files) != 1:
            raise RuntimeError(f"Expected exactly one {key} file in {calibration_folder}, found {len(files)}.")
        trials[key] = C3dData(files[0].as_posix())

    # --- Find the full name of each marker --- #
    markers = NexusMarkers(trials["static"])

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

    # LShank
    ltib = markers[Markers.LTIB]
    ltibf = markers[Markers.LTIBF]
    ltibd = markers[Markers.LTIBD]
    lank = markers[Markers.LANK]
    lankm = markers[Markers.LANKM]

    # LFoot
    lhee = markers[Markers.LHEE]
    lnav = markers[Markers.LNAV]
    ltoe = markers[Markers.LTOE]
    ltoe5 = markers[Markers.LTOE5]

    # RThigh
    rthi = markers[Markers.RTHI]
    rthib = markers[Markers.RTHIB]
    rthid = markers[Markers.RTHID]
    rknee = markers[Markers.RKNEE]
    rkneem = markers[Markers.RKNEEM]

    # RShank
    rtib = markers[Markers.RTIB]
    rtibf = markers[Markers.RTIBF]
    rtibd = markers[Markers.RTIBD]
    rank = markers[Markers.RANK]
    rankm = markers[Markers.RANKM]

    # RFoot
    rhee = markers[Markers.RHEE]
    rnav = markers[Markers.RNAV]
    rtoe = markers[Markers.RTOE]
    rtoe5 = markers[Markers.RTOE5]

    # --- Generate the personalized kinematic model --- #
    model = BiomechanicalModel()

    # Hip
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
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers([lpsi, rpsi]),
                    end=SegmentCoordinateSystemUtils.mean_markers([lasi, rasi]),
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh((lpsi, rpsi, rasi, lasi, lpsi), is_local=False),
        )
    )
    model.segments["Pelvis"].add_marker(Marker(lpsi, is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker(rpsi, is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker(lasi, is_technical=True, is_anatomical=True))
    model.segments["Pelvis"].add_marker(Marker(rasi, is_technical=True, is_anatomical=True))

    # LThigh
    lknee_mid = SegmentCoordinateSystemUtils.mean_markers([lknee, lkneem])
    lthi_origin = (
        SegmentCoordinateSystemUtils.score(
            functional_data=trials["left_hip_functionnal"],
            parent_marker_names=[lpsi, rpsi, lasi, rasi],
            child_marker_names=[lthi, lthib, lthid],
            vizualize=False,
        )
        if use_score
        else lasi
    )

    model.add_segment(
        Segment(
            name="LThigh",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lthi_origin,
                first_axis=Axis(name=Axis.Name.Z, start=lknee_mid, end=lthi_origin),
                second_axis=Axis(name=Axis.Name.X, start=lknee, end=lkneem),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((lasi, lthi, lthib, lthid, lthi, lthid, lknee_mid, lknee, lkneem), is_local=False),
        )
    )
    model.segments["LThigh"].add_marker(Marker(lthi, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthib, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthid, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lknee, is_technical=False, is_anatomical=True))
    model.segments["LThigh"].add_marker(Marker(lkneem, is_technical=False, is_anatomical=True))

    # LShank
    lank_mid = SegmentCoordinateSystemUtils.mean_markers([lank, lankm])
    shank_vertical_axis = Axis(name=Axis.Name.Z, start=lank_mid, end=lknee_mid)
    ltib_axis, ltib_origin = (
        SegmentCoordinateSystemUtils.sara(
            name=Axis.Name.X,
            functional_data=trials["left_knee_functionnal"],
            parent_marker_names=[lthi, lthib, lthid],
            child_marker_names=[ltib, ltibf, ltibd],
            perpendicular_axis=shank_vertical_axis,
            vizualize=True,
        )
        # if use_score
        # else (Axis(name=Axis.Name.X, start=lknee, end=lkneem), lknee_mid)
    )
    model.add_segment(
        Segment(
            name="LShank",
            parent_name="LThigh",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=ltib_origin,
                first_axis=shank_vertical_axis,
                second_axis=ltib_axis,
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh((ltibd, ltib, ltibf, ltibd, lank_mid, lank, lankm), is_local=False),
        )
    )
    model.segments["LShank"].add_marker(Marker(ltib, is_technical=True, is_anatomical=False))
    model.segments["LShank"].add_marker(Marker(ltibf, is_technical=True, is_anatomical=False))
    model.segments["LShank"].add_marker(Marker(ltibd, is_technical=True, is_anatomical=False))
    model.segments["LShank"].add_marker(Marker(lank, is_technical=False, is_anatomical=True))
    model.segments["LShank"].add_marker(Marker(lankm, is_technical=False, is_anatomical=True))

    # LFoot
    lfoot_origin = (
        SegmentCoordinateSystemUtils.score(
            functional_data=trials["left_ankle_functionnal"],
            parent_marker_names=[ltib, ltibf, ltibd],
            child_marker_names=[lhee, lnav, ltoe, ltoe5],
            vizualize=True,
        )
        if use_score
        else lank_mid
    )
    model.add_segment(
        Segment(
            name="LFoot",
            parent_name="LShank",
            rotations=Rotations.XZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lfoot_origin,
                first_axis=Axis(Axis.Name.Z, start=ltoe, end=lhee),
                second_axis=Axis(Axis.Name.X, start=lank, end=lankm),
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh((lhee, lnav, ltoe, lhee, ltoe, ltoe5, lhee, ltoe5, lnav), is_local=False),
        )
    )
    model.segments["LFoot"].add_marker(Marker(lhee, is_technical=True, is_anatomical=True))
    model.segments["LFoot"].add_marker(Marker(lnav, is_technical=True, is_anatomical=True))
    model.segments["LFoot"].add_marker(Marker(ltoe, is_technical=True, is_anatomical=True))
    model.segments["LFoot"].add_marker(Marker(ltoe5, is_technical=True, is_anatomical=True))

    # RThigh
    rknee_mid = SegmentCoordinateSystemUtils.mean_markers([rknee, rkneem])
    rthi_origin = (
        SegmentCoordinateSystemUtils.score(
            functional_data=trials["right_hip_functionnal"],
            parent_marker_names=[lpsi, rpsi, lasi, rasi],
            child_marker_names=[rthi, rthib, rthid],
            vizualize=False,
        )
        if use_score
        else rasi
    )
    model.add_segment(
        Segment(
            name="RThigh",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=rthi_origin,
                first_axis=Axis(name=Axis.Name.Z, start=rknee_mid, end=rthi_origin),
                second_axis=Axis(name=Axis.Name.X, start=rknee, end=rkneem),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((rasi, rthi, rthib, rthid, rthi, rthid, rknee_mid, rknee, rkneem), is_local=False),
        )
    )
    model.segments["RThigh"].add_marker(Marker(rthi, is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker(rthib, is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker(rthid, is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker(rknee, is_technical=False, is_anatomical=True))
    model.segments["RThigh"].add_marker(Marker(rkneem, is_technical=False, is_anatomical=True))

    # RShank
    rank_mid = SegmentCoordinateSystemUtils.mean_markers([rank, rankm])
    rtib_origin = rknee_mid
    model.add_segment(
        Segment(
            name="RShank",
            parent_name="RThigh",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=rtib_origin,
                first_axis=Axis(name=Axis.Name.Z, start=rank_mid, end=rtib_origin),
                second_axis=Axis(name=Axis.Name.X, start=rknee, end=rkneem),
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh((rtibd, rtib, rtibf, rtibd, rank_mid, rank, rankm), is_local=False),
        )
    )
    model.segments["RShank"].add_marker(Marker(rtib, is_technical=True, is_anatomical=False))
    model.segments["RShank"].add_marker(Marker(rtibf, is_technical=True, is_anatomical=False))
    model.segments["RShank"].add_marker(Marker(rtibd, is_technical=True, is_anatomical=False))
    model.segments["RShank"].add_marker(Marker(rank, is_technical=False, is_anatomical=True))
    model.segments["RShank"].add_marker(Marker(rankm, is_technical=False, is_anatomical=True))

    # RFoot
    rfoot_origin = (
        SegmentCoordinateSystemUtils.score(
            functional_data=trials["right_ankle_functionnal"],
            parent_marker_names=[rtib, rtibf, rtibd],
            child_marker_names=[rhee, rnav, rtoe, rtoe5],
            vizualize=True,
        )
        if use_score
        else rank_mid
    )
    model.add_segment(
        Segment(
            name="RFoot",
            parent_name="RShank",
            rotations=Rotations.XZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=rfoot_origin,
                first_axis=Axis(Axis.Name.Z, start=rtoe, end=rhee),
                second_axis=Axis(Axis.Name.X, start=rankm, end=rank),
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh((rhee, rnav, rtoe, rhee, rtoe, rtoe5, rhee, rtoe5, rnav), is_local=False),
        )
    )
    model.segments["RFoot"].add_marker(Marker(rhee, is_technical=True, is_anatomical=False))
    model.segments["RFoot"].add_marker(Marker(rnav, is_technical=True, is_anatomical=False))
    model.segments["RFoot"].add_marker(Marker(rtoe, is_technical=True, is_anatomical=False))
    model.segments["RFoot"].add_marker(Marker(rtoe5, is_technical=True, is_anatomical=False))

    # Put the model together, print it and print it to a bioMod file
    model_real = model.to_real(trials["static"])
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
        output_model_filepath=output_model_filepath,
        calibration_folder=calibration_folder,
        animate_model=True,
        use_score=False,
    )


if __name__ == "__main__":
    main()
