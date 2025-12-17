from enum import Enum
import logging
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
    SegmentCoordinateSystemUtils,
    BiomechanicalModelReal,
)

from ..utils.nexus_markers import NexusMarkers

_logger = logging.getLogger(__name__)


class Markers(Enum):

    C7 = "C7"
    C2 = "C2"
    T6 = "T6"
    T10 = "T10"
    S1 = "S1"
    S3 = "S3"
    CLAV = "CLAV"
    STRN = "STRN"

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


def generate_lower_body_model(calibration_folder: Path, use_score: bool = True) -> BiomechanicalModelReal:
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
    static_trial = NexusMarkers(trials["static"], expected_marker_names=tuple(Markers))

    # Trunk
    c7 = static_trial[Markers.C7]
    c2 = static_trial[Markers.C2]
    t6 = static_trial[Markers.T6]
    t10 = static_trial[Markers.T10]
    s1 = static_trial[Markers.S1]
    s3 = static_trial[Markers.S3]
    clav = static_trial[Markers.CLAV]
    strn = static_trial[Markers.STRN]

    # Hip
    lpsi = static_trial[Markers.LPSI]
    rpsi = static_trial[Markers.RPSI]
    lasi = static_trial[Markers.LASI]
    rasi = static_trial[Markers.RASI]

    # LThigh
    lthi = static_trial[Markers.LTHI]
    lthib = static_trial[Markers.LTHIB]
    lthid = static_trial[Markers.LTHID]
    lknee = static_trial[Markers.LKNEE]
    lkneem = static_trial[Markers.LKNEEM]

    # LShank
    ltib = static_trial[Markers.LTIB]
    ltibf = static_trial[Markers.LTIBF]
    ltibd = static_trial[Markers.LTIBD]
    lank = static_trial[Markers.LANK]
    lankm = static_trial[Markers.LANKM]

    # LFoot
    lhee = static_trial[Markers.LHEE]
    lnav = static_trial[Markers.LNAV]
    ltoe = static_trial[Markers.LTOE]
    ltoe5 = static_trial[Markers.LTOE5]

    # RThigh
    rthi = static_trial[Markers.RTHI]
    rthib = static_trial[Markers.RTHIB]
    rthid = static_trial[Markers.RTHID]
    rknee = static_trial[Markers.RKNEE]
    rkneem = static_trial[Markers.RKNEEM]

    # RShank
    rtib = static_trial[Markers.RTIB]
    rtibf = static_trial[Markers.RTIBF]
    rtibd = static_trial[Markers.RTIBD]
    rank = static_trial[Markers.RANK]
    rankm = static_trial[Markers.RANKM]

    # RFoot
    rhee = static_trial[Markers.RHEE]
    rnav = static_trial[Markers.RNAV]
    rtoe = static_trial[Markers.RTOE]
    rtoe5 = static_trial[Markers.RTOE5]

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

    # Trunk
    model.add_segment(
        Segment(
            name="Trunk",
            parent_name="Pelvis",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=clav,
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers([t10, c7]),
                    end=SegmentCoordinateSystemUtils.mean_markers([strn, clav]),
                ),
                second_axis=Axis(
                    name=Axis.Name.Z,
                    start=SegmentCoordinateSystemUtils.mean_markers([t10, strn]),
                    end=SegmentCoordinateSystemUtils.mean_markers([c7, clav]),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((s3, s1, t10, t6, c7, c2, c7, clav, strn, t10), is_local=False),
        )
    )
    model.segments["Trunk"].add_marker(Marker(c7, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(c2, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(t6, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(t10, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(s1, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(s3, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(clav, is_technical=True, is_anatomical=True))
    model.segments["Trunk"].add_marker(Marker(strn, is_technical=True, is_anatomical=True))

    # LThigh
    lknee_mid = SegmentCoordinateSystemUtils.mean_markers([lknee, lkneem])
    lthi_origin = (
        SegmentCoordinateSystemUtils.score(
            functional_data=trials["left_hip_functionnal"],
            parent_marker_names=[lpsi, rpsi, lasi, rasi],
            child_marker_names=[lthi, lthib, lthid],
            visualize=False,
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
                second_axis=Axis(name=Axis.Name.X, start=lasi, end=rasi),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (lthi_origin, lthi, lthib, lthi_origin, lthib, lthid, lthi, lthid, lknee_mid, lknee, lkneem),
                is_local=False,
            ),
        )
    )
    model.segments["LThigh"].add_marker(Marker(lthi_origin, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthi, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthib, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthid, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lknee, is_technical=False, is_anatomical=True))
    model.segments["LThigh"].add_marker(Marker(lkneem, is_technical=False, is_anatomical=True))

    # LShank
    ltib_axis = (
        SegmentCoordinateSystemUtils.sara(
            name=Axis.Name.X,
            functional_data=trials["left_knee_functionnal"],
            parent_marker_names=[ltibd, ltib, ltibf],  # Child and parent swapped to get correct axis direction
            child_marker_names=[lthib, lthid, lthi],
            visualize=False,
        )
        if use_score
        else Axis(name=Axis.Name.X, start=lknee_mid, end=lkneem)
    )
    lank_mid = SegmentCoordinateSystemUtils.mean_markers([lank, lankm])
    model.add_segment(
        Segment(
            name="LShank",
            parent_name="LThigh",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=ltib_axis.start,
                first_axis=Axis(name=Axis.Name.Z, start=lank_mid, end=ltib_axis.start),
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
            visualize=False,
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
            visualize=False,
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
                second_axis=Axis(name=Axis.Name.X, start=lasi, end=rasi),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (rthi_origin, rthi, rthib, rthi_origin, rthib, rthid, rthi, rthid, rknee_mid, rknee, rkneem),
                is_local=False,
            ),
        )
    )
    model.segments["RThigh"].add_marker(Marker(rthi, is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker(rthib, is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker(rthid, is_technical=True, is_anatomical=False))
    model.segments["RThigh"].add_marker(Marker(rknee, is_technical=False, is_anatomical=True))
    model.segments["RThigh"].add_marker(Marker(rkneem, is_technical=False, is_anatomical=True))

    # RShank
    rtib_axis = (
        SegmentCoordinateSystemUtils.sara(
            name=Axis.Name.X,
            functional_data=trials["right_knee_functionnal"],
            parent_marker_names=[rthid, rthi, rthib],
            child_marker_names=[rtib, rtibf, rtibd],
            visualize=False,
        )
        if use_score
        else Axis(name=Axis.Name.X, start=rkneem, end=rknee_mid)
    )
    rank_mid = SegmentCoordinateSystemUtils.mean_markers([rank, rankm])
    model.add_segment(
        Segment(
            name="RShank",
            parent_name="RThigh",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=rtib_axis.start,
                first_axis=Axis(name=Axis.Name.Z, start=rank_mid, end=rtib_axis.start),
                second_axis=rtib_axis,
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
            visualize=False,
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

    _logger.info("Collapsing the model to real...")
    model_real = model.to_real(trials["static"])
    return model_real
