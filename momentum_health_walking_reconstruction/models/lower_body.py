from enum import Enum
import logging
from pathlib import Path

from biobuddy import (
    Axis,
    BiomechanicalModel,
    Marker,
    Mesh,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    SegmentCoordinateSystemUtils,
    BiomechanicalModelReal,
    DeLevaTable,
    Sex,
    SegmentName,
)
import numpy as np

from ..utils.data_markers import DataMarkers

_logger = logging.getLogger(__name__)


def _compute_mean_marker_height(trial: DataMarkers, marker_names: list[str]) -> float:
    return float(np.nanmean(trial.filter(expected_marker_names=marker_names).to_numpy(), axis=2).mean(axis=1)[2])


def _compute_mean_length_between_markers(trial: DataMarkers, marker_name_1: str, marker_name_2: str) -> float:
    markers = trial.filter(expected_marker_names=[marker_name_1, marker_name_2]).to_numpy()[:3, :, :]
    return float(np.nanmean((((markers[:, 0, :] - markers[:, 1, :]) ** 2).sum(axis=0) ** 0.5)))


class Markers(Enum):
    LFHD = "LFHD"
    RFHD = "RFHD"
    LBHD = "LBHD"
    RBHD = "RBHD"

    LSHO = "LSHO"
    LELB = "LELB"
    LWRA = "LWRA"
    LWRB = "LWRB"
    LFIN = "LFIN"
    RSHO = "RSHO"
    RELB = "RELB"
    RWRA = "RWRA"
    RWRB = "RWRB"
    RFIN = "RFIN"

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


# Head
lfhd = Markers.LFHD.value
rfhd = Markers.RFHD.value
lbhd = Markers.LBHD.value
rbhd = Markers.RBHD.value

# Arms
lsho = Markers.LSHO.value
lelb = Markers.LELB.value
lwra = Markers.LWRA.value
lwrb = Markers.LWRB.value
lfin = Markers.LFIN.value
rsho = Markers.RSHO.value
relb = Markers.RELB.value
rwra = Markers.RWRA.value
rwrb = Markers.RWRB.value
rfin = Markers.RFIN.value

# Trunk
c7 = Markers.C7.value
c2 = Markers.C2.value
t6 = Markers.T6.value
t10 = Markers.T10.value
s1 = Markers.S1.value
s3 = Markers.S3.value
clav = Markers.CLAV.value
strn = Markers.STRN.value

# Hip
lpsi = Markers.LPSI.value
rpsi = Markers.RPSI.value
lasi = Markers.LASI.value
rasi = Markers.RASI.value

# LThigh
lthi = Markers.LTHI.value
lthib = Markers.LTHIB.value
lthid = Markers.LTHID.value
lknee = Markers.LKNEE.value
lkneem = Markers.LKNEEM.value
# LShank
ltib = Markers.LTIB.value
ltibf = Markers.LTIBF.value
ltibd = Markers.LTIBD.value
lank = Markers.LANK.value
lankm = Markers.LANKM.value
# LFoot
lhee = Markers.LHEE.value
lnav = Markers.LNAV.value
ltoe = Markers.LTOE.value
ltoe5 = Markers.LTOE5.value

# RThigh
rthi = Markers.RTHI.value
rthib = Markers.RTHIB.value
rthid = Markers.RTHID.value
rknee = Markers.RKNEE.value
rkneem = Markers.RKNEEM.value
# RShank
rtib = Markers.RTIB.value
rtibf = Markers.RTIBF.value
rtibd = Markers.RTIBD.value
rank = Markers.RANK.value
rankm = Markers.RANKM.value
# RFoot
rhee = Markers.RHEE.value
rnav = Markers.RNAV.value
rtoe = Markers.RTOE.value
rtoe5 = Markers.RTOE5.value


def generate_lower_body_model(calibration_folder: Path, use_score: bool = True) -> BiomechanicalModelReal:
    # --- Load all the required data files --- #
    trial_names = {
        "static": ["*func_anat.c3d", tuple([m.value for m in Markers])],
        "left_hip_functionnal": ["*func_lhip.c3d", (lpsi, rpsi, lasi, rasi, lthi, lthib, lthid)],
        "left_knee_functionnal": ["*func_lknee.c3d", (lthi, lthib, lthid, ltib, ltibf, ltibd, lknee, lkneem)],
        "left_ankle_functionnal": ["*func_lankle.c3d", (ltib, ltibf, ltibd, lhee, lnav, ltoe, ltoe5)],
        "right_hip_functionnal": ["*func_rhip.c3d", (lpsi, rpsi, lasi, rasi, rthi, rthib, rthid)],
        "right_knee_functionnal": ["*func_rknee.c3d", (rthi, rthib, rthid, rtib, rtibf, rtibd, rknee, rkneem)],
        "right_ankle_functionnal": ["*func_rankle.c3d", (rtib, rtibf, rtibd, rhee, rnav, rtoe, rtoe5)],
    }

    trials: dict[str, DataMarkers] = {}
    for key, values in trial_names.items():
        pattern = values[0]
        expected_marker_names = values[1]
        files = list(calibration_folder.glob(pattern))
        if len(files) != 1:
            raise RuntimeError(f"Expected exactly one {key} file in {calibration_folder}, found {len(files)}.")
        try:
            trials[key] = DataMarkers.from_c3d(files[0]).filter(
                expected_marker_names=expected_marker_names, rename_markers=False
            )
        except Exception as e:
            raise RuntimeError(f"Error while loading '{files[0].name}' trial: {e}") from e

    # --- Generate the personalized kinematic model --- #
    model = BiomechanicalModel()

    # --- Dynamic model --- #
    height = _compute_mean_marker_height(trials["static"], [lfhd, rfhd, lbhd, rbhd])
    de_leva_table = DeLevaTable(total_mass=100, sex=Sex.MALE)
    de_leva_table.from_measurements(
        total_height=height,
        ankle_height=_compute_mean_marker_height(trials["static"], [rank, rankm, lank, lankm]),
        knee_height=_compute_mean_marker_height(trials["static"], [rknee, rkneem, lknee, lkneem]),
        hip_height=_compute_mean_marker_height(trials["static"], [lthi, rthi]),
        shoulder_height=_compute_mean_marker_height(trials["static"], [lsho, rsho]),
        finger_span=_compute_mean_length_between_markers(trials["static"], lfin, rfin),
        wrist_span=_compute_mean_length_between_markers(trials["static"], lwra, rwra),
        elbow_span=_compute_mean_length_between_markers(trials["static"], lelb, relb),
        shoulder_span=_compute_mean_length_between_markers(trials["static"], lsho, rsho),
        foot_length=_compute_mean_length_between_markers(trials["static"], lhee, ltoe),
        hip_width=_compute_mean_length_between_markers(trials["static"], lthi, rthi),
    )
    # Change main axis of the foot to Z
    trunk_center_of_mass_function = de_leva_table[SegmentName.TRUNK].center_of_mass
    de_leva_table[SegmentName.TRUNK].center_of_mass = lambda m, bio: [
        0,
        0,
        trunk_center_of_mass_function(m, bio)[2]
        - (
            np.nanmean(m.values["CLAV"], axis=1)[2]
            - bio.segments["Pelvis"].segment_coordinate_system.scs.translation[2]
        ),
    ]
    foot_center_of_mass_function = de_leva_table[SegmentName.FOOT].center_of_mass
    de_leva_table[SegmentName.FOOT].center_of_mass = lambda m, bio: -1 * foot_center_of_mass_function(m, bio)[[2, 1, 0]]

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
            inertia_parameters=de_leva_table[SegmentName.TRUNK],
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

    # Left arm
    model.add_segment(
        Segment(
            name="LUpperArm",
            parent_name="Trunk",
            translations=Translations.XYZ,
            rotations=Rotations.XY,
            inertia_parameters=de_leva_table[SegmentName.UPPER_ARM],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lsho,
                first_axis=Axis(Axis.Name.Z, start=lelb, end=lsho),
                second_axis=Axis(Axis.Name.X, start=lwrb, end=lwra),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((lsho, lelb), is_local=False),
        )
    )
    model.segments["LUpperArm"].add_marker(Marker(lsho, is_technical=True, is_anatomical=False))
    model.segments["LUpperArm"].add_marker(Marker(lelb, is_technical=True, is_anatomical=False))

    model.add_segment(
        Segment(
            name="LLowerArm",
            parent_name="LUpperArm",
            rotations=Rotations.XYZ,
            inertia_parameters=de_leva_table[SegmentName.LOWER_ARM],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lelb,
                first_axis=Axis(Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers([lwra, lwrb]), end=lelb),
                second_axis=Axis(Axis.Name.X, start=lwrb, end=lwra),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((lelb, lwra, lwrb, lelb), is_local=False),
        )
    )
    model.segments["LLowerArm"].add_marker(Marker(lwra, is_technical=True, is_anatomical=False))
    model.segments["LLowerArm"].add_marker(Marker(lwrb, is_technical=True, is_anatomical=False))

    # Right arm
    model.add_segment(
        Segment(
            name="RUpperArm",
            parent_name="Trunk",
            translations=Translations.XYZ,
            rotations=Rotations.XY,
            inertia_parameters=de_leva_table[SegmentName.UPPER_ARM],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=rsho,
                first_axis=Axis(Axis.Name.Z, start=relb, end=rsho),
                second_axis=Axis(Axis.Name.X, start=rwrb, end=rwra),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((rsho, relb), is_local=False),
        )
    )
    model.segments["RUpperArm"].add_marker(Marker(rsho, is_technical=True, is_anatomical=False))
    model.segments["RUpperArm"].add_marker(Marker(relb, is_technical=True, is_anatomical=False))

    model.add_segment(
        Segment(
            name="RLowerArm",
            parent_name="RUpperArm",
            rotations=Rotations.XYZ,
            inertia_parameters=de_leva_table[SegmentName.LOWER_ARM],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=relb,
                first_axis=Axis(Axis.Name.Z, start=SegmentCoordinateSystemUtils.mean_markers([rwra, rwrb]), end=relb),
                second_axis=Axis(Axis.Name.X, start=rwrb, end=rwra),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((relb, rwra, rwrb, relb), is_local=False),
        )
    )
    model.segments["RLowerArm"].add_marker(Marker(rwra, is_technical=True, is_anatomical=False))
    model.segments["RLowerArm"].add_marker(Marker(rwrb, is_technical=True, is_anatomical=False))

    # LThigh
    lknee_mid = SegmentCoordinateSystemUtils.mean_markers([lknee, lkneem])
    lthi_origin = (
        SegmentCoordinateSystemUtils.score(
            functional_data=trials["left_hip_functionnal"].to_biobuddy(),
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
            rotations=Rotations.XZY,
            inertia_parameters=de_leva_table[SegmentName.THIGH],
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
    model.segments["LThigh"].add_marker(Marker(lthi, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthib, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lthid, is_technical=True, is_anatomical=False))
    model.segments["LThigh"].add_marker(Marker(lknee, is_technical=False, is_anatomical=True))
    model.segments["LThigh"].add_marker(Marker(lkneem, is_technical=False, is_anatomical=True))

    # LShank
    ltib_axis = (
        SegmentCoordinateSystemUtils.sara(
            name=Axis.Name.X,
            functional_data=trials["left_knee_functionnal"].to_biobuddy(),
            parent_marker_names=[ltibd, ltib, ltibf],  # Child and parent swapped to get correct axis direction
            child_marker_names=[lthib, lthid, lthi],
            expected_rotation_axis_orientation=Axis("AoR", start=lknee, end=lkneem),
            origin_positions_global=lambda x, model: (x.values[lknee] + x.values[lkneem]) / 2,
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
            inertia_parameters=de_leva_table[SegmentName.SHANK],
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
            functional_data=trials["left_ankle_functionnal"].to_biobuddy(),
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
            inertia_parameters=de_leva_table[SegmentName.FOOT],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lfoot_origin,
                first_axis=Axis(Axis.Name.Z, start=ltoe, end=lhee),
                second_axis=Axis(Axis.Name.X, start=lank, end=lankm),
                axis_to_keep=Axis.Name.Z,
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
            functional_data=trials["right_hip_functionnal"].to_biobuddy(),
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
            rotations=Rotations.XZY,
            inertia_parameters=de_leva_table[SegmentName.THIGH],
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
            functional_data=trials["right_knee_functionnal"].to_biobuddy(),
            parent_marker_names=[rthid, rthi, rthib],
            child_marker_names=[rtib, rtibf, rtibd],
            expected_rotation_axis_orientation=Axis("AoR", start=rkneem, end=rknee),
            origin_positions_global=lambda x, model: (x.values[rknee] + x.values[rkneem]) / 2,
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
            inertia_parameters=de_leva_table[SegmentName.SHANK],
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
            functional_data=trials["right_ankle_functionnal"].to_biobuddy(),
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
            inertia_parameters=de_leva_table[SegmentName.FOOT],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=rfoot_origin,
                first_axis=Axis(Axis.Name.Z, start=rtoe, end=rhee),
                second_axis=Axis(Axis.Name.X, start=rankm, end=rank),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((rhee, rnav, rtoe, rhee, rtoe, rtoe5, rhee, rtoe5, rnav), is_local=False),
        )
    )
    model.segments["RFoot"].add_marker(Marker(rhee, is_technical=True, is_anatomical=False))
    model.segments["RFoot"].add_marker(Marker(rnav, is_technical=True, is_anatomical=False))
    model.segments["RFoot"].add_marker(Marker(rtoe, is_technical=True, is_anatomical=False))
    model.segments["RFoot"].add_marker(Marker(rtoe5, is_technical=True, is_anatomical=False))

    _logger.info("Collapsing the model to real...")
    model_real = model.to_real(trials["static"].to_biobuddy())

    # Use Pelvis as root segment
    model_real.segments["Pelvis"].parent_name = model_real.segments["root"].parent_name
    model_real.segments["Pelvis"].segment_coordinate_system = model_real.segments["root"].segment_coordinate_system
    model_real.segments._remove("root")
    return model_real
