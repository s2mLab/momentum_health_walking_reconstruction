from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks


if TYPE_CHECKING:
    from ..comparison_analyses.kinematics_data import KinematicsData, Side, Point
from .math import find_first_below_threshold, derivative


class SwayTrial:
    def __init__(
        self,
        starting_index_in_data: int,
        center_of_mass_data: np.ndarray,
        frame_rate: int,
    ):
        self._starting_index_in_data = starting_index_in_data
        self._center_of_mass_data = center_of_mass_data
        self._frame_rate = frame_rate

    @classmethod
    def from_data(
        cls,
        center_of_mass_data: np.ndarray,
        frame_rate: int,
        starting_index_in_data: int,
    ):
        return cls(
            starting_index_in_data=starting_index_in_data,
            center_of_mass_data=center_of_mass_data,
            frame_rate=frame_rate,
        )

    def length(self, exclude_vertical: bool = False) -> int:
        data_slice = slice(None) if not exclude_vertical else slice(2)
        return np.sum(np.linalg.norm(np.diff(self._center_of_mass_data[data_slice, :], axis=1), axis=0))

    def amplitude(self, exclude_vertical: bool = False) -> np.ndarray:
        data_slice = slice(None) if not exclude_vertical else slice(2)
        return np.ptp(self._center_of_mass_data[data_slice, :], axis=1)

    def velocity(self, exclude_vertical: bool = False) -> np.ndarray:
        data_slice = slice(None) if not exclude_vertical else slice(2)
        return np.linalg.norm(derivative(self._center_of_mass_data[data_slice, :], frame_rate=self._frame_rate), axis=0)

    def confidence_ellipse(self, confidence_level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        # Compute the covariance matrix of the center of mass data
        covariance_matrix = np.cov(self._center_of_mass_data[:2, :])

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Compute the angle of the ellipse
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        # Compute the width and height of the ellipse based on the confidence level
        chi_square_value = np.sqrt(-2 * np.log(1 - confidence_level))
        width = 2 * chi_square_value * np.sqrt(eigenvalues[0])
        height = 2 * chi_square_value * np.sqrt(eigenvalues[1])

        return (width, height), angle

    @classmethod
    def extract(cls, kinematics_data: KinematicsData, show_plot: bool = False) -> SwayTrial:
        from ..comparison_analyses.kinematics_data import Point

        center_of_mass_data = kinematics_data.points(point=Point.CENTER_OF_MASS)

        indices = _get_trial_indices(
            expected_duration=25.0,
            frame_rate=kinematics_data.frame_rate(),
            center_of_mass_data=center_of_mass_data,
            minimum_velocity_threshold=0.5,
            minimum_zeros_frame=30,
            show_plot=show_plot,
        )
        return SwayTrial.from_data(
            center_of_mass_data=center_of_mass_data[:, indices[0] : indices[1]],
            frame_rate=kinematics_data.frame_rate(),
            starting_index_in_data=indices[0],
        )


def _get_trial_indices(
    expected_duration: float,
    frame_rate: int,
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
    frame_rate : int
        The frame rate of the data, used to convert the indices into frames.
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
    com_velocity: np.ndarray = np.linalg.norm(derivative(center_of_mass_data, frame_rate=frame_rate), axis=0)

    # Two peaks are expected (as it is absolute velocity), one for the squat and one for the standing up
    squat_peaks = find_peaks(com_velocity, height=com_velocity.max() * 0.5)[0]
    if len(squat_peaks) != 2:
        raise ValueError(
            f"Expected 2 peaks in the center of mass velocity, but found {len(squat_peaks)}. Peaks found at indices: {squat_peaks}"
        )
    peak = squat_peaks[1]

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
    expected_cycle_frame_count = int(expected_duration * frame_rate)
    end = start_index + expected_cycle_frame_count

    if show_plot:
        time_vector = np.arange(len(com_velocity)) / frame_rate
        plt.figure()
        plt.plot(time_vector, com_velocity, label=f"Center of Mass Velocity")
        # Plot the cycles as shaded areas
        plt.axvspan(start_index / frame_rate, end / frame_rate, color="gray", alpha=0.3)
        plt.legend()
        plt.title(f"Center of Mass Velocity with Identified Trial")
        plt.xlabel("Frame")
        plt.ylabel("Velocity (m/s)")
        plt.show()

    return (start_index, end)
