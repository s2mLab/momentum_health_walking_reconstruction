from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

import biorbd
import ezc3d
from matplotlib import pyplot as plt
import numpy as np
import struct
import pygltflib
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from scipy.signal import correlate, find_peaks

from ..utils.math import find_first_below_threshold, derivative, nanunwrap

if TYPE_CHECKING:
    from ..utils.analyses_data import GaitCycle, SwayTrial


class TrialType(Enum):
    GAIT = auto()
    SWAY = auto()


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()
    NOT_SIDED = auto()


class Joint(Enum):
    TRUNK = auto()
    PELVIS = auto()
    HIP = auto()
    KNEE = auto()
    ANKLE = auto()


class Point(Enum):
    CENTER_OF_MASS = auto()
    LEFT_HEEL = auto()
    LEFT_TOE = auto()
    RIGHT_HEEL = auto()
    RIGHT_TOE = auto()


class KinematicsData(ABC):
    def __init__(self, original_frame_rate: float, last_frame_index: int, trial_type: TrialType):
        self._original_frame_rate = original_frame_rate
        self._resample_ratio = 1
        self._initial_frame_index = 0
        self._original_last_frame_index = last_frame_index
        self._last_frame_index = last_frame_index
        self._trial_type = trial_type

    def duplicate_alignment_data_from(self, reference: KinematicsData):
        if isinstance(self, EmptyKinematicsData) or isinstance(reference, EmptyKinematicsData):
            return

        if (
            self._original_frame_rate != reference._original_frame_rate
            or self._trial_type != reference._trial_type
            or self._original_last_frame_index != reference._original_last_frame_index
        ):
            raise ValueError(
                "Cannot duplicate align data from another KinematicsData with different original frame rate, trial type, or original last frame index."
            )

        self.set_resample_ratio(reference.resample_ratio(resampled=True))
        self.set_initial_frame_index(0)
        self.set_frame_count(reference.frame_count(resampled=True))
        self.set_initial_frame_index(reference.initial_frame_index(resampled=True))

    @property
    def trial_type(self) -> TrialType:
        return self._trial_type

    def time_vector(self, resampled: bool = True) -> np.ndarray:
        full_time_vector = np.arange(self.frame_count(resampled=False)) / self.frame_rate(resampled=False)
        return full_time_vector[:: self.resample_ratio(resampled=resampled)]

    def frame_count(self, resampled: bool = True) -> int:
        frame_count = (
            self.last_frame_index(resampled=False) - self.initial_frame_index(resampled=False) + 1
        ) / self.resample_ratio(resampled=resampled)
        if frame_count - int(frame_count) > 0:
            return int(frame_count) + 1
        return int(frame_count)

    def frame_rate(self, resampled: bool = True) -> float:
        return self._original_frame_rate // self.resample_ratio(resampled=resampled)

    def resample_ratio(self, resampled: bool = True) -> int:
        return self._resample_ratio if resampled else 1

    def set_resample_ratio(self, ratio: int) -> None:
        if ratio < 1:
            raise ValueError("Resample ratio must be a positive integer.")

        if self._original_frame_rate % ratio != 0:
            raise ValueError("Resample ratio must be a divisor of the original frame rate.")
        self._resample_ratio = int(ratio)

    def initial_frame_index(self, resampled: bool = True) -> int:
        return self._initial_frame_index // self.resample_ratio(resampled=resampled)

    def set_initial_frame_index(self, new_frame_index: int):
        new_frame_index_in_original = new_frame_index * self.resample_ratio(resampled=True)

        if new_frame_index_in_original < 0 or new_frame_index_in_original >= self._original_last_frame_index:
            raise ValueError("Offset must be a non-negative integer.")
        self._initial_frame_index = new_frame_index_in_original

    def last_frame_index(self, resampled: bool = True) -> int:
        return self._last_frame_index // self.resample_ratio(resampled=resampled)

    def original_last_frame_index(self, resampled: bool = True) -> int:
        return self._original_last_frame_index // self.resample_ratio(resampled=resampled)

    def set_frame_count(self, new_frame_count: int):
        initial_frame_index = self.initial_frame_index(resampled=False)
        new_last_frame_index_in_original = (
            new_frame_count * self.resample_ratio(resampled=True) + initial_frame_index - 1
        )

        if new_last_frame_index_in_original < 0 or new_last_frame_index_in_original >= self._original_last_frame_index:
            raise ValueError("Offset must be a non-negative integer.")
        self._last_frame_index = new_last_frame_index_in_original

    def _data_slice(self, resampled: bool = True) -> slice:
        initial_frame_index = self.initial_frame_index(resampled=False)
        last_frame_index = self.last_frame_index(resampled=False) + 1
        resample_ratio = self.resample_ratio(resampled=resampled)
        return slice(initial_frame_index, last_frame_index, resample_ratio)

    @abstractmethod
    def angles(self, joint: Joint, side: Side, resampled: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def points(self, point: Point, resampled: bool = True) -> np.ndarray:
        """
        Returns:
            A (3, N) array containing the 3D coordinates of the specified point across the frames.
        """
        pass

    def points_velocity(self, point: Point, resampled: bool = True) -> np.ndarray:
        return derivative(self.points(point, resampled=resampled), frame_rate=self.frame_rate(resampled=resampled))

    @abstractmethod
    def extract_gait_cycles(self, side: Side, show_plot: bool = False) -> list[GaitCycle]:
        pass

    @abstractmethod
    def extract_sway_trial(self, show_plot: bool = False) -> SwayTrial:
        pass

    def plot(
        self,
        joint: Joint,
        side: Side,
        resampled: bool = True,
        title: str = None,
        label: str = None,
        show_now: bool = True,
    ) -> None:
        if title is not None:
            plt.figure(title)

        time_vector = self.time_vector(resampled=resampled)
        plt.plot(
            time_vector,
            self.angles(joint=joint, side=side, resampled=resampled),
            label=f"{side.name} {joint.name}" + (f" - {label}" if label is not None else ""),
        )

        plt.legend()
        plt.title("Joint angles")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (radian)")

        if show_now:
            plt.show()

    @staticmethod
    def perform_align_kinematics_data(data1: KinematicsData, data2: KinematicsData, show_plot: bool = False) -> None:
        if data1.frame_count() == 0 or data2.frame_count() == 0:
            return

        if data1.frame_rate(resampled=False) > data2.frame_rate(resampled=False):
            data1.set_resample_ratio(data1.frame_rate(resampled=False) // data2.frame_rate())
        elif data2.frame_count() > data1.frame_count():
            data2.set_resample_ratio(data2.frame_rate(resampled=False) // data1.frame_rate())

        data1_com_position = data1.points(point=Point.CENTER_OF_MASS, resampled=True)[2, :]
        data2_com_position = data2.points(point=Point.CENTER_OF_MASS, resampled=True)[2, :]
        if show_plot:
            title = "Center of mass vertical position before alignment"
            plt.figure(title)
            plt.plot(data1.time_vector(resampled=True), data1_com_position, "r-", label="Data1")
            plt.plot(data2.time_vector(resampled=True), data2_com_position, "b-", label="Data2")
            plt.legend()
            plt.title(title)
            plt.xlabel("Time (s)")
            plt.ylabel("Height (m)")

        # Normalize the height by aligning the mean together
        data1_com_position -= data1_com_position.mean()
        data2_com_position -= data2_com_position.mean()

        corr = correlate(data1_com_position, data2_com_position, mode="full")
        lags = np.arange(-len(data2_com_position) + 1, len(data1_com_position))
        best_lag = lags[np.argmax(corr)]

        if best_lag > 0:
            data1.set_initial_frame_index(data1.initial_frame_index() + best_lag)
        elif best_lag < 0:
            data2.set_initial_frame_index(data2.initial_frame_index() - best_lag)

        data1_remaining_frames = data1.original_last_frame_index() - data1.initial_frame_index()
        data2_remaining_frames = data2.original_last_frame_index() - data2.initial_frame_index()
        if data1_remaining_frames > data2_remaining_frames:
            data1.set_frame_count(data2_remaining_frames)
        elif data2_remaining_frames > data1_remaining_frames:
            data2.set_frame_count(data1_remaining_frames)

        if show_plot:
            title = "Center of mass vertical position after alignment"
            plt.figure(title)
            plt.plot(
                data1.time_vector(resampled=True),
                data1.points(point=Point.CENTER_OF_MASS, resampled=True)[2, :],
                "r-",
                label="Data1",
            )
            plt.plot(
                data2.time_vector(resampled=True),
                data2.points(point=Point.CENTER_OF_MASS, resampled=True)[2, :],
                "b-",
                label="Data2",
            )
            plt.legend()
            plt.title(title)
            plt.xlabel("Time (s)")
            plt.ylabel("Height (m)")
            plt.show()


class EmptyKinematicsData(KinematicsData):
    def __init__(self, trial_type: TrialType):
        self._data = {}
        super().__init__(
            original_frame_rate=0.0,
            last_frame_index=-1,
            trial_type=trial_type,
        )

    def angles(self, joint: Joint, side: Side, resampled: bool = True) -> np.ndarray:
        return np.ndarray((0,))

    def points(self, point: Point, resampled: bool = True) -> np.ndarray:
        return np.ndarray((3, 0))

    def extract_gait_cycles(self, side: Side, show_plot: bool = False) -> list[GaitCycle]:
        return []

    def extract_sway_trial(self, show_plot: bool = False) -> SwayTrial:
        return None


class BiorbdKinematicsData(KinematicsData):
    def __init__(self, model: biorbd.Biorbd, c3d_data: ezc3d.c3d, kinematics: np.ndarray, trial_type: TrialType):
        self._model = model

        # Load the inhouse model data along with the data used to compute the kinematics
        self._c3d = c3d_data
        self._kinematics: np.ndarray = kinematics

        super().__init__(
            original_frame_rate=self._c3d.header["points"]["frame_rate"],
            last_frame_index=kinematics.shape[1] - 1,
            trial_type=trial_type,
        )

    def _get_kinematics(self, resampled: bool = True) -> np.ndarray:
        return self._kinematics[:, self._data_slice(resampled=resampled)]

    def _get_points_data(self, resampled: bool = True) -> np.ndarray:
        return self._c3d.data["points"][:3, :, self._data_slice(resampled=resampled)] / 1000.0

    @property
    def point_names(self) -> list[str]:
        return self._c3d.parameters["POINT"]["LABELS"]["value"]

    def angles(self, joint: Joint, side: Side, resampled: bool = True) -> np.ndarray:
        if side == Side.LEFT:
            side_prefix = "L"
        elif side == Side.RIGHT:
            side_prefix = "R"
        elif side == Side.NOT_SIDED:
            side_prefix = ""
        else:
            raise ValueError("Invalid side. Must be Side.LEFT, Side.RIGHT, or Side.NOT_SIDED.")

        if joint == Joint.TRUNK:
            joint_prefix = "Trunk_RotX"
            side_prefix = ""
            multiplier = 1
            offset = 0
        elif joint == Joint.PELVIS:
            joint_prefix = "Pelvis_RotX"
            side_prefix = ""
            multiplier = 1
            offset = 0
        elif joint == Joint.HIP:
            if side != Side.LEFT and side != Side.RIGHT:
                raise ValueError("Hip joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_prefix = "Thigh_RotX"
            multiplier = 1  # Make Flexion positive
            offset = 0
        elif joint == Joint.KNEE:
            if side != Side.LEFT and side != Side.RIGHT:
                raise ValueError("Knee joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_prefix = "Shank_RotX"
            multiplier = -1  # Make Flexion positive
            offset = 0
        elif joint == Joint.ANKLE:
            if side != Side.LEFT and side != Side.RIGHT:
                raise ValueError("Ankle joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_prefix = "Foot_RotX"
            multiplier = -1  # Make Dorsiflexion positive
            offset = np.pi / 2
        else:
            raise ValueError("Unsupported joint. Only TRUNK, PELVIS, HIP, KNEE, and ANKLE are currently supported.")

        joint_index = self._model.dof_names.index(f"{side_prefix}{joint_prefix}")
        return self._get_kinematics(resampled=resampled)[joint_index, :] * multiplier + offset

    def points(self, point: Point, resampled: bool = True) -> np.ndarray:
        if point == Point.CENTER_OF_MASS:
            q = self._get_kinematics(resampled=resampled)
            frame_count = self.frame_count(resampled=resampled)
            return np.array([self._model.center_of_mass(q[:, index]) for index in range(frame_count)]).T
        elif point == Point.LEFT_HEEL:
            left_heel_index = self._find_marker_index(self.point_names, suffix="LHEE")
            return self._get_points_data(resampled=resampled)[:3, left_heel_index, :]
        elif point == Point.RIGHT_HEEL:
            right_heel_index = self._find_marker_index(self.point_names, suffix="RHEE")
            return self._get_points_data(resampled=resampled)[:3, right_heel_index, :]
        elif point == Point.LEFT_TOE:
            left_toe_index = self._find_marker_index(self.point_names, suffix="LTOE5")
            return self._get_points_data(resampled=resampled)[:3, left_toe_index, :]
        elif point == Point.RIGHT_TOE:
            right_toe_index = self._find_marker_index(self.point_names, suffix="RTOE5")
            return self._get_points_data(resampled=resampled)[:3, right_toe_index, :]
        else:
            raise ValueError(
                "Unsupported point. Only CENTER_OF_MASS, LEFT_HEEL, RIGHT_HEEL, LEFT_TOE, and RIGHT_TOE are currently supported."
            )

    @classmethod
    def from_file(
        cls, model_path: str, c3d_path: str, kinematics_path: str, trial_type: TrialType
    ) -> BiorbdKinematicsData:
        if kinematics_path is None:
            return EmptyKinematicsData(trial_type=trial_type)

        model = biorbd.Biorbd(model_path)

        # Load the inhouse model data along with the data used to compute the kinematics
        c3d_data = ezc3d.c3d(str(c3d_path))
        kinematics = np.load(kinematics_path, allow_pickle=True)

        return cls(model=model, c3d_data=c3d_data, kinematics=kinematics, trial_type=trial_type)

    @staticmethod
    def _find_marker_index(marker_names: list[str], suffix: str) -> int:
        """Find the index of a marker in the marker names list based on a suffix."""
        for i, name in enumerate(marker_names):
            if name.endswith(suffix):
                return i
        raise ValueError(f"Marker with suffix '{suffix}' not found in marker names.")

    def extract_gait_cycles(self, side: Side, show_plot: bool = False) -> list[GaitCycle]:
        from ..utils.analyses_data import GaitCycle

        if self._trial_type != TrialType.GAIT:
            raise ValueError("Gait cycles can only be extracted from GAIT trials.")

        if side == Side.LEFT:
            heel_point = Point.LEFT_HEEL
            toe_point = Point.LEFT_TOE
        elif side == Side.RIGHT:
            heel_point = Point.RIGHT_HEEL
            toe_point = Point.RIGHT_TOE
        else:
            raise ValueError("Unsupported side. Only LEFT and RIGHT are currently supported.")

        heel_data = self.points(point=heel_point)
        toe_data = self.points(point=toe_point)
        center_of_mass_data = self.points(point=Point.CENTER_OF_MASS)

        cycle_indices = self._get_gait_cycle_indices(
            expect_cycle_duration=1.0,
            heel_data=heel_data,
            toe_data=toe_data,
            maximum_peak_threshold=0.75,
            minimum_velocity_threshold=0.5,
            minimum_zeros_frame=10,
            show_plot=show_plot,
            side=side,
        )
        cycles: list[GaitCycle] = []
        for indices in cycle_indices:
            cycles.append(
                GaitCycle(
                    toe_off_index=indices[1] - indices[0],
                    center_of_mass_data=center_of_mass_data[:, indices[0] : indices[2]],
                    heel_data=heel_data[:, indices[0] : indices[2]],
                    frame_rate=self.frame_rate(),
                    starting_index_in_data=indices[0],
                )
            )
        return cycles

    def extract_sway_trial(self, show_plot: bool = False) -> SwayTrial:
        from ..utils.analyses_data import SwayTrial

        if self._trial_type != TrialType.SWAY:
            raise ValueError("Sway trials can only be extracted from SWAY trials.")

        center_of_mass_data = self.points(point=Point.CENTER_OF_MASS)

        indices = self._get_sway_trial_indices(
            expected_duration=25.0,
            center_of_mass_data=center_of_mass_data,
            minimum_velocity_threshold=0.5,
            minimum_zeros_frame=30,
            show_plot=show_plot,
        )
        return SwayTrial(
            center_of_mass_data=center_of_mass_data[:, indices[0] : indices[1]],
            frame_rate=self.frame_rate(),
            starting_index_in_data=indices[0],
        )

    def _get_gait_cycle_indices(
        self,
        expect_cycle_duration: float,
        heel_data: np.ndarray,
        toe_data: np.ndarray,
        maximum_peak_threshold: float,
        minimum_velocity_threshold: float,
        minimum_zeros_frame: int,
        show_plot: bool,
        side: Side,
    ) -> list[tuple[int, int, int]]:
        """
        Get gait cycles from data using heel strikes and toe offs.
        - Heel strikes are identified as the first almost no velocity after maxima in the velocity of the heel marker.
        - Toe offs are identified as the first almost no velocity before maxima in the velocity of the toe marker,
        but only if it happens after a heel strike and before the next heel strike.
        - Gait cycles are defined as starting with a heel strike and ending with the next heel strike, containing one toe off in between.
        and the duration of the cycle is between half and double the expected cycle duration.

        Parameters
        ----------
        expect_cycle_duration : float
            The expected duration of a gait cycle in seconds, used to set the minimum distance between peaks.
        heel_data : np.ndarray
            The 3D positions of the heel marker across frames, used to compute velocity and identify heel strikes (shape: [num_frames, 3]).
        toe_data : np.ndarray
            The 3D positions of the toe marker across frames, used to compute velocity and identify toe offs (shape: [num_frames, 3]).
        maximum_peak_threshold : float, optional
            The minimum height of the peaks in the heel velocity to be considered valid, as a fraction of the maximum velocity, by default 0.75.
        minimum_velocity_threshold : float, optional
            The velocity threshold below which a point is considered a heel strike or toe off, by default 0.5 m/s.
        minimum_zeros_frame : int, optional
            The minimum number of consecutive frames with velocity below the threshold to confirm a heel strike or toe off, by default 10 frames.
        show_plot : bool, optional
            Whether to show a plot of the heel velocity with identified events, by default False.
        side : Side
            The side of the body (LEFT or RIGHT) for which to identify gait cycles (this is for plot purposes).

        Returns
        -------
        list[tuple[int, int, int]]
            A list of tuples, each containing the frame indices of (heel_strike, toe_off, next_heel_strike) for each identified gait cycle.
        """
        # Aliases
        expected_cycle_frame_count = int(expect_cycle_duration * self.frame_rate())

        # Find the middle of the swing phase by finding the peaks in the velocity of the heel marker
        heel_velocity: np.ndarray = np.linalg.norm(derivative(heel_data, self.frame_rate()), axis=0)
        mid_swing_peaks = find_peaks(
            heel_velocity,
            height=np.nanmax(heel_velocity) * maximum_peak_threshold,
            distance=int(expected_cycle_frame_count // 2),
        )[0]

        # Heel strikes are identified as the first almost no velocity after the peaks in the velocity of the heel marker
        heel_strikes = []
        for peak in mid_swing_peaks:
            heel_strike = find_first_below_threshold(
                velocity_data=heel_velocity,
                start_index=peak,
                threshold=minimum_velocity_threshold,
                minimum_frame_count=minimum_zeros_frame,
                direction=1,
            )
            if heel_strike != -1:
                heel_strikes.append(heel_strike)

        # Find toe offs using the velocity of the toe marker
        ltoe_velocity: np.ndarray = np.linalg.norm(derivative(toe_data, self.frame_rate()), axis=0)

        toe_offs = []
        for heel_strike in mid_swing_peaks:
            toe_off = find_first_below_threshold(
                velocity_data=ltoe_velocity,
                start_index=heel_strike,
                threshold=minimum_velocity_threshold,
                minimum_frame_count=minimum_zeros_frame,
                direction=-1,
            )
            if toe_off != -1:
                toe_offs.append(toe_off)

        # Cycles starts with a heel strike and ends with the next heel strike and contains one toe off
        gait_cycles = []
        for i in range(len(heel_strikes) - 1):
            start = heel_strikes[i]
            end = heel_strikes[i + 1]
            toe_off = next((t for t in toe_offs if start < t < end), None)
            if (
                toe_off is not None
                and (end - start) >= expected_cycle_frame_count // 2
                and (end - start) <= expected_cycle_frame_count * 2
            ):
                gait_cycles.append((start, toe_off, end))

        if show_plot:
            plt.figure()
            plt.plot(heel_velocity, label=f"{side.name} Heel")
            plt.plot(ltoe_velocity, label=f"{side.name} Toe")
            plt.plot(mid_swing_peaks, heel_velocity[mid_swing_peaks], "o", label="Peaks")
            plt.plot(heel_strikes, np.zeros_like(heel_velocity[heel_strikes]), "x", label="Heel Strikes")
            plt.plot(toe_offs, np.zeros_like(ltoe_velocity[toe_offs]), "s", label="Toe Offs")
            # Plot the cycles as shaded areas
            for start, toe_off, end in gait_cycles:
                plt.axvspan(start, end, color="gray", alpha=0.3)
            plt.legend()
            plt.title(f"{side.name} Heel trajectory")
            plt.xlabel("Frame")
            plt.ylabel("Velocity (m/s)")
            plt.show()

        return gait_cycles

    def _get_sway_trial_indices(
        self,
        expected_duration: float,
        center_of_mass_data: np.ndarray,
        minimum_velocity_threshold: float,
        minimum_zeros_frame: int,
        show_plot: bool,
    ) -> tuple[int, int]:
        """
        Get the indices of the standing still trial. It assumes a squat was perform just prior to the trial and that it last
        at least expected_duration seconds.

        Parameters
        ----------
        expected_duration : float
            The expected duration of the standing still trial in seconds. Used to compute the end index
        center_of_mass_data : np.ndarray
            The 3D positions of the center of mass marker across frames, used to compute velocity and identify events (shape: [num_frames, 3]).
        minimum_velocity_threshold : float, optional
            The velocity threshold below which a point is considered a valid event, by default 0.5 m/s.
        minimum_zeros_frame : int, optional
            The minimum number of consecutive frames with velocity below the threshold to confirm a valid event, by default 10 frames.
        show_plot : bool, optional
            Whether to show a plot of the center of mass velocity with identified events, by default False.

        Returns
        -------
        tuple[int, int]
            A tuple containing the frame indices of (start, end) for the identified trial.
        """
        # Find the middle of the swing phase by finding the peaks in the velocity of the center of mass marker
        com_velocity: np.ndarray = np.linalg.norm(derivative(center_of_mass_data, frame_rate=self.frame_rate()), axis=0)

        # Two peaks are expected (as it is absolute velocity), one for the squat and one for the standing up
        squat_peaks = find_peaks(com_velocity, height=com_velocity.max() * 0.15, distance=15)[0]
        peak = squat_peaks[-1]

        # Starting is when the squat is over
        start_index = find_first_below_threshold(
            velocity_data=com_velocity,
            start_index=peak,
            threshold=minimum_velocity_threshold,
            minimum_frame_count=minimum_zeros_frame,
            direction=1,
        )
        start_index += minimum_zeros_frame  # Make sure we are after the squat

        # Ending is when the trial is over, which is expected_duration seconds after the start
        expected_cycle_frame_count = int(expected_duration * self.frame_rate())
        end = start_index + expected_cycle_frame_count

        if show_plot:
            time_vector = np.arange(len(com_velocity)) / self.frame_rate()
            plt.figure()
            plt.plot(time_vector, com_velocity, label=f"Center of Mass Velocity")
            # Plot the cycles as shaded areas
            plt.axvspan(start_index / self.frame_rate(), end / self.frame_rate(), color="gray", alpha=0.3)
            plt.legend()
            plt.title(f"Center of Mass Velocity with Identified Trial")
            plt.xlabel("Frame")
            plt.ylabel("Velocity (m/s)")
            plt.show()

        return (start_index, end)


class MomentumHealthGlbKinematicsData(KinematicsData):
    def __init__(self, data: dict[str, np.ndarray], trial_type: TrialType):
        self._data = data
        super().__init__(
            original_frame_rate=30.0,
            last_frame_index=next(iter(data.values())).shape[0] - 1,
            trial_type=trial_type,
        )

    def set_resample_ratio(self, ratio: int) -> None:
        raise NotImplementedError(
            "Resampling is not supported for MomentumHealth GLB data since it is already at a fixed frame rate of 30 FPS."
        )

    def angles(self, joint: Joint, side: Side, resampled: bool = True) -> np.ndarray:
        if side == Side.LEFT:
            hip_name = "left_hip"
            knee_name = "left_knee"
            ankle_name = "left_ankle"
        elif side == Side.RIGHT:
            hip_name = "right_hip"
            knee_name = "right_knee"
            ankle_name = "right_ankle"
        else:
            raise ValueError("Invalid side. Must be Side.LEFT or Side.RIGHT.")

        if joint == Joint.KNEE:
            data_slice = self._data_slice(resampled=resampled)
            hip_to_knee = self._data[knee_name][data_slice, :] - self._data[hip_name][data_slice, :]
            knee_to_ankle = self._data[ankle_name][data_slice, :] - self._data[knee_name][data_slice, :]
            hip_to_knee_norm = hip_to_knee / (np.linalg.norm(hip_to_knee, axis=1, keepdims=True) + 1e-6)
            knee_to_ankle_norm = knee_to_ankle / (np.linalg.norm(knee_to_ankle, axis=1, keepdims=True) + 1e-6)
            return np.arccos(np.clip(np.sum(hip_to_knee_norm * knee_to_ankle_norm, axis=1), -1, 1))
        else:
            raise ValueError("Unsupported joint. Only KNEE is currently supported.")

    def points(self, point: Point, resampled: bool = True) -> np.ndarray:
        if point == Point.CENTER_OF_MASS:
            raise NotImplementedError("Center of mass data is not available in the MomentumHealth GLB file.")
        else:
            raise ValueError("Unsupported point. Only CENTER_OF_MASS is currently supported.")

    @classmethod
    def from_file(cls, glb_path: str, trial_type: TrialType) -> MomentumHealthGlbKinematicsData:
        gltf = pygltflib.GLTF2.load(glb_path)

        node_names = {}
        node_parents = {}
        for i, node in enumerate(gltf.nodes):
            node_names[i] = node.name
            if node.children:
                for child_idx in node.children:
                    node_parents[child_idx] = i

        if not gltf.animations:
            print("No animations found in GLB")
            return None

        anim = gltf.animations[0]

        def read_accessor(accessor_idx):
            accessor = gltf.accessors[accessor_idx]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            start = buffer_view.byteOffset + accessor.byteOffset

            if accessor.componentType == 5126:  # FLOAT
                fmt_char = "f"
                bytes_per_elem = 4
            else:
                raise ValueError(f"Unsupported component type: {accessor.componentType}")

            if accessor.type == "SCALAR":
                num_components = 1
            elif accessor.type == "VEC3":
                num_components = 3
            elif accessor.type == "VEC4":
                num_components = 4
            else:
                raise ValueError(f"Unsupported type: {accessor.type}")

            stride = buffer_view.byteStride if buffer_view.byteStride else num_components * bytes_per_elem
            values = []
            current = start
            for _ in range(accessor.count):
                chunk = data[current : current + num_components * bytes_per_elem]
                val = struct.unpack(f"<{num_components}{fmt_char}", chunk)
                values.append(val if num_components > 1 else val[0])
                current += stride
            return np.array(values)

        node_channels = {}
        samplers = []
        for s in anim.samplers:
            times = read_accessor(s.input)
            values = read_accessor(s.output)
            samplers.append({"times": times, "values": values, "interpolation": s.interpolation})

        for ch in anim.channels:
            node_idx = ch.target.node
            path = ch.target.path
            if node_idx not in node_channels:
                node_channels[node_idx] = {}
            node_channels[node_idx][path] = samplers[ch.sampler]

        all_times = samplers[0]["times"]
        num_frames = len(all_times)

        base_transforms = {}
        for i, node in enumerate(gltf.nodes):
            trans = np.array(node.translation) if node.translation else np.array([0.0, 0.0, 0.0])
            rot = np.array(node.rotation) if node.rotation else np.array([0.0, 0.0, 0.0, 1.0])
            scale = np.array(node.scale) if node.scale else np.array([1.0, 1.0, 1.0])
            base_transforms[i] = (trans, rot, scale)

        frame_data = []
        for f in tqdm(range(num_frames), desc="Processing frames"):
            local_transforms = {}
            for i in range(len(gltf.nodes)):
                trans, rot, scale = base_transforms[i]
                if i in node_channels:
                    if "translation" in node_channels[i]:
                        trans = node_channels[i]["translation"]["values"][f]
                    if "rotation" in node_channels[i]:
                        rot = node_channels[i]["rotation"]["values"][f]
                    if "scale" in node_channels[i]:
                        scale = node_channels[i]["scale"]["values"][f]

                r = Rotation.from_quat(rot).as_matrix()
                T = np.eye(4)
                T[:3, :3] = r * scale
                T[:3, 3] = trans
                local_transforms[i] = T

            global_transforms = {}

            def get_global_transform(node_idx):
                if node_idx in global_transforms:
                    return global_transforms[node_idx]
                local_T = local_transforms[node_idx]
                if node_idx in node_parents:
                    parent_T = get_global_transform(node_parents[node_idx])
                    global_T = parent_T @ local_T
                else:
                    global_T = local_T
                global_transforms[node_idx] = global_T
                return global_T

            frame_positions = {}
            for i in range(len(gltf.nodes)):
                T = get_global_transform(i)
                name = node_names[i]
                if name:
                    frame_positions[name] = T[:3, 3]
            frame_data.append(frame_positions)

        glb_pos = {}
        for frame in frame_data:
            for name, pos in frame.items():
                if name not in glb_pos:
                    glb_pos[name] = []
                glb_pos[name].append(pos)
        for name in glb_pos:
            glb_pos[name] = np.array(glb_pos[name])
        return cls(data=glb_pos, trial_type=trial_type)


class MomentumHealthCsvKinematicsData(KinematicsData):
    def __init__(
        self, data: dict[str, np.ndarray], trial_type: TrialType, precomputed_metrics: dict[str, list], frame_count: int
    ):
        self._data = data
        self._precomputed_metrics = precomputed_metrics
        super().__init__(
            original_frame_rate=60.0,
            last_frame_index=frame_count - 1,
            trial_type=trial_type,
        )

    def angles(self, joint: Joint, side: Side, resampled: bool = True) -> np.ndarray:
        if side == Side.LEFT:
            side_suffix = "_L"
        elif side == Side.RIGHT:
            side_suffix = "_R"
        elif side == Side.NOT_SIDED:
            side_suffix = ""
        else:
            raise ValueError("Invalid side. Must be Side.LEFT, Side.RIGHT, or Side.NOT_SIDED.")

        data_slice = self._data_slice(resampled=resampled)
        if joint == Joint.TRUNK:
            joint_name = "trunk_lean_sagittal_deg"
            side_suffix = ""  # The trunk does not have a side suffix
            multiplier = 1
            offset = 0
        elif joint == Joint.PELVIS:
            joint_name = "pelvis_rotation_deg"
            side_suffix = ""  # The pelvis does not have a side suffix
            multiplier = 1
            offset = 0
        elif joint == Joint.HIP:
            if side != Side.LEFT and side != Side.RIGHT:
                raise ValueError("Hip joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_name = "hip_flexion_deg"
            multiplier = 1  # Make Flexion positive
            offset = 0
        elif joint == Joint.KNEE:
            if side != Side.LEFT and side != Side.RIGHT:
                raise ValueError("Knee joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_name = "knee_flexion_deg"
            multiplier = -1  # Make Flexion positive
            offset = -180  # The data are 180 degrees offset
        elif joint == Joint.ANKLE:
            if side != Side.LEFT and side != Side.RIGHT:
                raise ValueError("Ankle joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_name = "ankle_dorsiflexion_deg"
            multiplier = 1  # Make Flexion positive
            offset = 0
        else:
            raise ValueError("Unsupported joint. Only TRUNK, PELVIS, HIP, KNEE, and ANKLE are currently supported.")

        return nanunwrap((self._data[joint_name + side_suffix][data_slice] + offset) * np.pi / 180.0 * multiplier)

    def points(self, point: Point, resampled: bool = True) -> np.ndarray:
        data_slice = self._data_slice(resampled=resampled)

        if point == Point.CENTER_OF_MASS:
            return np.concatenate(
                [[self._data["pelvis_z"]], [self._data["pelvis_x"]], [self._data["pelvis_y"]]], axis=0
            )[:, data_slice]
        else:
            raise ValueError("Unsupported point. Only CENTER_OF_MASS is currently supported.")

    def extract_gait_cycles(self, side: Side, show_plot: bool = False) -> list[GaitCycle]:
        from ..utils.analyses_data import PreComputedGaitCycle, GaitCycle

        if self._trial_type != TrialType.GAIT:
            raise ValueError("Gait cycles can only be extracted from GAIT trials.")

        out: list[GaitCycle] = []
        step_count = len(self._precomputed_metrics["step_index"])
        start_frame_index = None
        duration = None
        length = None
        double_support_time = None
        has_started_step = False
        for step_index in range(step_count):
            if self._precomputed_metrics["start_foot"][step_index] == side:
                start_frame_index = self._precomputed_metrics["start_frame"][step_index]
                duration = self._precomputed_metrics["duration_sec"][step_index]
                length = self._precomputed_metrics["length_m"][step_index]
                double_support_time = self._precomputed_metrics["double_support_sec"][step_index]
                has_started_step = True
            elif has_started_step and self._precomputed_metrics["end_foot"][step_index] == side:
                duration += self._precomputed_metrics["duration_sec"][step_index]
                length += self._precomputed_metrics["length_m"][step_index]
                double_support_time += self._precomputed_metrics["double_support_sec"][step_index]
                has_started_step = False

                out.append(
                    PreComputedGaitCycle(
                        stride_time=duration,
                        double_stance_time=double_support_time,
                        stride_length=length,
                        starting_index_in_data=start_frame_index,
                    )
                )

        # Remove what is definitely not a cycle (There is a lot of variability because of the participant stop walking)
        mean, std = GaitCycle.mean_gait_speed_from_cycles(out, compute_std=True)
        out = [
            cycle
            for cycle in out
            if cycle.mean_gait_speed() >= mean - std / 1.5 and cycle.mean_gait_speed() <= mean + std / 1.5
        ]
        return out

    def extract_sway_trial(self, show_plot: bool = False) -> SwayTrial:
        from ..utils.analyses_data import PreComputedSwayTrial

        if self._trial_type != TrialType.SWAY:
            raise ValueError("Sway trials can only be extracted from SWAY trials.")

        return PreComputedSwayTrial(
            antero_posterior_amplitude=self._precomputed_metrics["sway_rms_ap_cm"] / 100.0,
            medio_lateral_amplitude=self._precomputed_metrics["sway_rms_ml_cm"] / 100.0,
            mean_velocity=self._precomputed_metrics["sway_velocity_cm_s"] / 100.0,
            confidence_ellipse_area=self._precomputed_metrics["sway_ellipse_area_cm2"] / 100.0**2,
            length=self._precomputed_metrics["sway_path_length_cm"] / 100.0,
        )

    @classmethod
    def from_file(cls, csv_path: str, trial_type: TrialType) -> MomentumHealthCsvKinematicsData:
        if csv_path is None:
            return EmptyKinematicsData(trial_type=trial_type)

        with open(csv_path, "r") as f:
            lines = f.readlines()

        # Find the label starting of data rows
        starting_line_label = "Per-Frame Series"
        starting_line_index = None
        for i, line in enumerate(lines):
            if line.startswith(starting_line_label):
                starting_line_index = i + 1
                break
        else:
            raise ValueError(f"Could not find starting label '{starting_line_label}' in CSV file.")

        # Get the data
        header_line = lines[starting_line_index].strip()
        column_names = header_line.split(",")
        data = {name: [] for name in column_names}
        frame_count = 0
        for line in lines[starting_line_index + 1 :]:
            if line == "\n":
                # We reached the end of the data section
                break
            values = line.strip().split(",")
            for name, value in zip(column_names, values):
                if name == "is_non_walking":
                    # data[name].append(value == "true")
                    pass
                else:
                    data[name].append(float(value))
            frame_count += 1

        for name in data:
            data[name] = np.array(data[name])

        # Get the precomputed metrics
        if trial_type == TrialType.GAIT:
            starting_line_index += frame_count + 3
            header_line = lines[starting_line_index].strip()
            column_names = header_line.split(",")
            precomputed_metrics = {name: [] for name in column_names}
            for line in lines[starting_line_index + 1 :]:
                if line == "\n":
                    # We reached the end of the data section
                    break
                values = line.strip().split(",")
                for name, value in zip(column_names, values):
                    if name in ["step_index", "start_frame", "end_frame"]:
                        precomputed_metrics[name].append(int(value))
                    elif name in ["start_foot", "end_foot"]:
                        if value == "L":
                            value = Side.LEFT
                        elif value == "R":
                            value = Side.RIGHT
                        else:
                            raise ValueError(f"Invalid foot value: {value}")
                        precomputed_metrics[name].append(value)
                    else:
                        precomputed_metrics[name].append(float(value))
        elif trial_type == TrialType.SWAY:
            starting_line_index = 8
            precomputed_metrics = {}
            for line in lines[starting_line_index + 1 :]:
                if line == "\n":
                    # We reached the end of the data section
                    break
                name, value = line.strip().split(",")
                precomputed_metrics[name] = float(value)

        else:
            raise ValueError(f"Unsupported trial type: {trial_type}")

        return cls(data=data, trial_type=trial_type, precomputed_metrics=precomputed_metrics, frame_count=frame_count)


class PigKinematicsData(KinematicsData):
    def __init__(self, data: dict[str, np.ndarray], frame_rate: float, trial_type: TrialType):
        self._data = data
        super().__init__(
            original_frame_rate=frame_rate,
            last_frame_index=next(iter(data.values())).shape[1] - 1,
            trial_type=trial_type,
        )

    def angles(self, joint: Joint, side: Side, resampled: bool = True) -> np.ndarray:
        if side == Side.LEFT:
            side_prefix = "L"
        elif side == Side.RIGHT:
            side_prefix = "R"
        elif side == Side.NOT_SIDED:
            side_prefix = ""
        else:
            raise ValueError("Invalid side. Must be Side.LEFT, Side.RIGHT, or Side.NOT_SIDED.")

        if joint == Joint.TRUNK:
            if side == Side.NOT_SIDED:
                side_prefix = "L"  # The values have side in the file, but R and L are identical
            joint_prefix = "ThoraxAngles"
        elif joint == Joint.PELVIS:
            if side == Side.NOT_SIDED:
                side_prefix = "L"  # The values have side in the file, but R and L are identical
            joint_prefix = "PelvisAngles"
        elif joint == Joint.HIP:
            if side == Side.NOT_SIDED:
                raise ValueError("Hip joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_prefix = "HipAngles"
        elif joint == Joint.KNEE:
            if side == Side.NOT_SIDED:
                raise ValueError("Knee joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_prefix = "KneeAngles"
        elif joint == Joint.ANKLE:
            if side == Side.NOT_SIDED:
                raise ValueError("Ankle joint must be sided. Use Side.LEFT or Side.RIGHT.")
            joint_prefix = "AnkleAngles"
        else:
            raise ValueError("Unsupported joint. Only KNEE and TRUNK are currently supported.")

        data_slice = self._data_slice(resampled=resampled)
        data = nanunwrap((self._data[f"{side_prefix}{joint_prefix}"][0, data_slice]) * np.pi / 180.0)
        if joint == Joint.HIP:
            data += 0
        elif joint == Joint.KNEE:
            if np.nanmean(data) > np.pi - 0.2:
                data = data - np.pi
            elif np.nanmean(data) < -np.pi + 0.2:
                data = data + np.pi
        elif joint == Joint.ANKLE:
            data = -1 * data + np.pi / 2
        return data

    def points(self, point: Point, resampled: bool = True) -> np.ndarray:
        data_slice = self._data_slice(resampled=resampled)
        if point == Point.CENTER_OF_MASS:
            return (
                np.mean([self._data[point_name] for point_name in ["LASI", "RASI", "LPSI", "RPSI"]], axis=0)[
                    :, data_slice
                ]
                / 1000.0
            )
        elif point == Point.LEFT_HEEL:
            return self._data["LHEE"][:, data_slice] / 1000.0
        elif point == Point.RIGHT_HEEL:
            return self._data["RHEE"][:, data_slice] / 1000.0
        elif point == Point.LEFT_TOE:
            return self._data["LTOE"][:, data_slice] / 1000.0
        elif point == Point.RIGHT_TOE:
            return self._data["RTOE"][:, data_slice] / 1000.0
        else:
            raise ValueError(
                "Unsupported point. Only CENTER_OF_MASS, LEFT_HEEL, RIGHT_HEEL, LEFT_TOE, and RIGHT_TOE are currently supported."
            )

    def extract_gait_cycles(self, side: Side, show_plot: bool = False) -> list["GaitCycle"]:
        return BiorbdKinematicsData.extract_gait_cycles(self, side=side, show_plot=show_plot)

    def _get_gait_cycle_indices(
        self,
        expect_cycle_duration: float,
        heel_data: np.ndarray,
        toe_data: np.ndarray,
        maximum_peak_threshold: float,
        minimum_velocity_threshold: float,
        minimum_zeros_frame: int,
        show_plot: bool,
        side: Side,
    ) -> list[tuple[int, int, int]]:
        return BiorbdKinematicsData._get_gait_cycle_indices(
            self,
            expect_cycle_duration=expect_cycle_duration,
            heel_data=heel_data,
            toe_data=toe_data,
            maximum_peak_threshold=maximum_peak_threshold,
            minimum_velocity_threshold=minimum_velocity_threshold,
            minimum_zeros_frame=minimum_zeros_frame,
            show_plot=show_plot,
            side=side,
        )

    def _get_sway_trial_indices(
        self,
        expected_duration: float,
        center_of_mass_data: np.ndarray,
        minimum_velocity_threshold: float,
        minimum_zeros_frame: int,
        show_plot: bool,
    ) -> tuple[int, int]:
        return BiorbdKinematicsData._get_sway_trial_indices(
            self,
            expected_duration=expected_duration,
            center_of_mass_data=center_of_mass_data,
            minimum_velocity_threshold=minimum_velocity_threshold,
            minimum_zeros_frame=minimum_zeros_frame,
            show_plot=show_plot,
        )

    def extract_sway_trial(self, show_plot: bool = False) -> "SwayTrial":
        return BiorbdKinematicsData.extract_sway_trial(self, show_plot=show_plot)

    @classmethod
    def from_file(cls, c3d_path: str, min_last_frame_index: int, trial_type: TrialType) -> PigKinematicsData:
        if c3d_path is None:
            return EmptyKinematicsData(trial_type=trial_type)

        # Load the kinematics stored in the C3D file
        c3d_data = ezc3d.c3d(str(c3d_path))

        first_frame = c3d_data.header["points"]["first_frame"]
        data_last_frame = c3d_data.header["points"]["last_frame"]
        expected_last_frame = max(data_last_frame, min_last_frame_index)
        data = {}
        for col_index, point_name in enumerate(c3d_data.parameters["POINT"]["LABELS"]["value"]):
            data[point_name] = np.ndarray((3, expected_last_frame + 1)) * np.nan
            data[point_name][:, first_frame : data_last_frame + 1] = c3d_data.data["points"][:3, col_index, :]

        frame_rate = c3d_data.header["points"]["frame_rate"]
        return cls(data=data, frame_rate=frame_rate, trial_type=trial_type)
