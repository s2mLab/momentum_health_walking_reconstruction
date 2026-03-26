from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks


if TYPE_CHECKING:
    from ..comparison_analyses.kinematics_data import KinematicsData, Side, Point


class GaitCycle:
    def __init__(
        self,
        starting_index_in_data: int,
        toe_off_index: int,
        gait_speed: np.ndarray,
        stride_length: float,
        stance_time: float,
        swing_time: float,
    ):
        self._starting_index_in_data = starting_index_in_data
        self._toe_off_index = toe_off_index
        self._gait_speed = gait_speed
        self._stride_length = stride_length
        self._stance_time = stance_time
        self._swing_time = swing_time

    @classmethod
    def from_data(
        cls,
        toe_off_index: int,
        heel_data: np.ndarray,
        center_of_mass_data: np.ndarray,
        frame_rate: int,
        starting_index_in_data: int,
    ):
        gait_speed = np.gradient(center_of_mass_data, 1 / frame_rate, axis=1, edge_order=2)
        stride_length = np.linalg.norm(heel_data[:, -1] - heel_data[:, 0])

        stance_time = toe_off_index / frame_rate
        swing_time = (heel_data.shape[1] - toe_off_index) / frame_rate

        return cls(
            starting_index_in_data=starting_index_in_data,
            toe_off_index=toe_off_index,
            gait_speed=gait_speed,
            stride_length=stride_length,
            stance_time=stance_time,
            swing_time=swing_time,
        )

    def gait_speed(self, exclude_vertical: bool = False) -> np.ndarray:
        return self._gait_speed[:2, :] if exclude_vertical else self._gait_speed

    def mean_gait_speed(self, exclude_vertical: bool = False) -> float:
        return np.mean(np.linalg.norm(self.gait_speed(exclude_vertical=exclude_vertical), axis=0))

    @property
    def stride_length(self) -> float:
        return self._stride_length

    @property
    def stride_time(self) -> float:
        return self._stance_time + self._swing_time

    @property
    def stance_time(self) -> float:
        return self._stance_time

    @property
    def swing_time(self) -> float:
        return self._swing_time

    @classmethod
    def gait_cycles(cls, kinematics_data: KinematicsData, side: Side, show_plot: bool = False) -> list[GaitCycle]:
        from ..comparison_analyses.kinematics_data import Point, Side

        if side == Side.LEFT:
            heel_point = Point.LEFT_HEEL
            toe_point = Point.LEFT_TOE
        elif side == Side.RIGHT:
            heel_point = Point.RIGHT_HEEL
            toe_point = Point.RIGHT_TOE
        else:
            raise ValueError("Unsupported side. Only LEFT and RIGHT are currently supported.")

        heel_data = kinematics_data.points(point=heel_point)
        toe_data = kinematics_data.points(point=toe_point)
        center_of_mass_data = kinematics_data.points(point=Point.CENTER_OF_MASS)

        cycle_indices = _get_gait_cycle_indices(
            expect_cycle_duration=1.0,
            frame_rate=kinematics_data.frame_rate(),
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
                GaitCycle.from_data(
                    toe_off_index=indices[1] - indices[0],
                    heel_data=heel_data[indices[0] : indices[2], :],
                    center_of_mass_data=center_of_mass_data[indices[0] : indices[2], :],
                    frame_rate=kinematics_data.frame_rate(),
                    starting_index_in_data=indices[0],
                )
            )
        return cycles


def _get_gait_cycle_indices(
    expect_cycle_duration: float,
    frame_rate: int,
    heel_data: np.ndarray,
    toe_data: np.ndarray,
    maximum_peak_threshold: float,
    minimum_velocity_threshold: float,
    minimum_zeros_frame: int,
    show_plot: bool,
    side: Side,
) -> list[tuple[int, int, int]]:
    """
    Get gait cycles from C3D data using heel strikes and toe offs.
    - Heel strikes are identified as the first almost no velocity after maxima in the velocity of the heel marker.
    - Toe offs are identified as the first almost no velocity before maxima in the velocity of the toe marker,
    but only if it happens after a heel strike and before the next heel strike.
    - Gait cycles are defined as starting with a heel strike and ending with the next heel strike, containing one toe off in between.
    and the duration of the cycle is between half and double the expected cycle duration.

    Parameters
    ----------
    expect_cycle_duration : float
        The expected duration of a gait cycle in seconds, used to set the minimum distance between peaks.
    frame_rate : int
        The frame rate of the C3D data, used to convert the expected cycle duration into frames.
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
    expected_cycle_frame_count = int(expect_cycle_duration * frame_rate)

    # Find the middle of the swing phase by finding the peaks in the velocity of the heel marker
    heel_velocity: np.ndarray = np.linalg.norm(np.gradient(heel_data, 1 / frame_rate, axis=0, edge_order=2), axis=1)
    mid_swing_peaks = find_peaks(
        heel_velocity,
        height=heel_velocity.max() * maximum_peak_threshold,
        distance=int(expected_cycle_frame_count // 2),
    )[0]

    # Heel strikes are identified as the first almost no velocity after the peaks in the velocity of the heel marker
    heel_strikes = []
    for peak in mid_swing_peaks:
        heel_strike = _find_first_below_threshold(
            velocity_data=heel_velocity,
            start_index=peak,
            threshold=minimum_velocity_threshold,
            minimum_frame_count=minimum_zeros_frame,
            direction=1,
        )
        if heel_strike != -1:
            heel_strikes.append(heel_strike)

    # Find toe offs using the velocity of the toe marker
    ltoe_velocity: np.ndarray = np.linalg.norm(np.gradient(toe_data, 1 / frame_rate, axis=0, edge_order=2), axis=1)

    toe_offs = []
    for heel_strike in mid_swing_peaks:
        toe_off = _find_first_below_threshold(
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


def _find_first_below_threshold(
    velocity_data: np.ndarray,
    start_index: int,
    threshold: float,
    minimum_frame_count: int,
    direction: int = 1,
) -> int:
    """
    Find the first index in velocity_data starting from start_index where the velocity drops below a threshold

    Parameters
    ----------
    velocity_data : np.ndarray
        The velocity data to search through.
    start_index : int
        The index to start searching from.
    threshold : float
        The velocity threshold to compare against.
    minimum_frame_count : int
        The minimum number of consecutive frames below the threshold to confirm a valid event.
    direction : int, optional
        The direction to search in: 1 for forward, -1 for backward, by default 1.
    """
    if direction not in [1, -1]:
        raise ValueError("Direction must be 1 (forward) or -1 (backward)")

    buffer = 0
    offset = 0
    while True:
        current_index = start_index + (direction * offset)
        if current_index < 0:
            return -1

        current_velocity = velocity_data[current_index]
        if current_velocity > threshold:
            buffer = 0
        else:
            buffer += 1
            if buffer > minimum_frame_count:
                return current_index + (-1 * direction * buffer)
        offset += 1
