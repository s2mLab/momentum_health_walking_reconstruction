from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np

from .math import derivative


class AnalysesData(ABC):
    pass

    @abstractmethod
    def indices(self) -> tuple[int, int]:
        """
        Returns the starting and ending indices of the data in the original time series.
        """
        pass


class MeanSpeedAlgorithm(Enum):
    TOTAL_DISTANCE_OVER_TOTAL_TIME = auto()
    AVERAGE_INSTANTANEOUS_SPEED = auto()


class SwayDirection(Enum):
    ANTERO_POSTERIOR = auto()
    MEDIO_LATERAL = auto()
    HORIZONTAL_PLANE = auto()


class GaitMetrics(Enum):
    GAIT_SPEED = auto()
    STRIDE_LENGTH = auto()
    STRIDE_TIME = auto()


class GaitCycle(AnalysesData):
    def __init__(
        self,
        toe_off_index: int,
        center_of_mass_data: np.ndarray,
        heel_data: np.ndarray,
        frame_rate: int,
        starting_index_in_data: int,
    ):
        self._starting_index_in_data = starting_index_in_data
        self._toe_off_index = toe_off_index
        self._center_of_mass_data = center_of_mass_data
        self._heel_data = heel_data
        self._frame_rate = frame_rate

    def indices(self) -> tuple[int, int]:
        return self._starting_index_in_data, self._starting_index_in_data + self._center_of_mass_data.shape[1]

    def gait_speed(self, exclude_vertical: bool = False) -> np.ndarray:
        data = self._center_of_mass_data[:2, :] if exclude_vertical else self._center_of_mass_data
        return derivative(data, frame_rate=self._frame_rate)

    @staticmethod
    def mean_gait_speed_from_cycles(
        cycles: list[GaitCycle],
        exclude_vertical: bool = False,
        algorithm: MeanSpeedAlgorithm = MeanSpeedAlgorithm.TOTAL_DISTANCE_OVER_TOTAL_TIME,
        compute_std: bool = False,
    ) -> float | tuple[float, float]:
        if not cycles:
            return np.nan if not compute_std else (np.nan, np.nan)
        out = np.mean(
            [cycle.mean_gait_speed(exclude_vertical=exclude_vertical, algorithm=algorithm) for cycle in cycles]
        )
        if compute_std:
            std = np.std(
                [cycle.mean_gait_speed(exclude_vertical=exclude_vertical, algorithm=algorithm) for cycle in cycles]
            )
            out = (out, std)

        return out

    def mean_gait_speed(
        self,
        exclude_vertical: bool = False,
        algorithm: MeanSpeedAlgorithm = MeanSpeedAlgorithm.TOTAL_DISTANCE_OVER_TOTAL_TIME,
    ) -> float:
        if algorithm == MeanSpeedAlgorithm.TOTAL_DISTANCE_OVER_TOTAL_TIME:
            return np.linalg.norm(self._center_of_mass_data[:, -1] - self._center_of_mass_data[:, 0]) / self.stride_time
        elif algorithm == MeanSpeedAlgorithm.AVERAGE_INSTANTANEOUS_SPEED:
            return np.mean(np.linalg.norm(self.gait_speed(exclude_vertical=exclude_vertical), axis=0))
        else:
            raise ValueError("Unsupported mean speed algorithm.")

    def stride_length(self, exclude_vertical: bool = False) -> float:
        data = self._heel_data[:2, :] if exclude_vertical else self._heel_data
        return np.linalg.norm(data[:, -1] - data[:, 0])

    @staticmethod
    def mean_stride_length_from_cycles(cycles: list[GaitCycle], exclude_vertical: bool = False) -> float:
        if not cycles:
            return np.nan
        return np.mean([cycle.stride_length(exclude_vertical=exclude_vertical) for cycle in cycles])

    @property
    def stride_time(self) -> float:
        return self.stance_time + self.swing_time

    @staticmethod
    def mean_stride_time_from_cycles(cycles: list[GaitCycle]) -> float:
        if not cycles:
            return np.nan
        return np.mean([cycle.stride_time for cycle in cycles])

    @property
    def stance_time(self) -> float:
        return self._toe_off_index / self._frame_rate

    @staticmethod
    def mean_stance_time_from_cycles(cycles: list[GaitCycle]) -> float:
        if not cycles:
            return np.nan
        return np.mean([cycle.stance_time for cycle in cycles])

    @property
    def swing_time(self) -> float:
        return (self._heel_data.shape[1] - self._toe_off_index) / self._frame_rate

    @staticmethod
    def mean_swing_time_from_cycles(cycles: list[GaitCycle]) -> float:
        if not cycles:
            return np.nan
        return np.mean([cycle.swing_time for cycle in cycles])


class PreComputedGaitCycle(AnalysesData):
    def __init__(
        self,
        stride_time: np.ndarray,
        double_stance_time: np.ndarray,
        stride_length: np.ndarray,
        starting_index_in_data: int,
    ):
        self._stride_time = stride_time
        self._double_stance_time = double_stance_time
        self._stride_length = stride_length
        self._starting_index_in_data = starting_index_in_data

    def indices(self) -> tuple[int, int]:
        raise ValueError(
            "PreComputedGaitCycle does not have a well-defined range of indices in the original time series."
        )

    def gait_speed(self, exclude_vertical: bool = False) -> np.ndarray:
        if exclude_vertical:
            raise ValueError("Vertical component is not available for PreComputedGaitCycle. Cannot exclude vertical.")
        raise ValueError("Gait speed is not available for PreComputedGaitCycle. Use mean_gait_speed instead.")

    def mean_gait_speed(
        self,
        exclude_vertical: bool = False,
        algorithm: MeanSpeedAlgorithm = MeanSpeedAlgorithm.TOTAL_DISTANCE_OVER_TOTAL_TIME,
    ) -> float:
        if exclude_vertical:
            raise ValueError("Vertical component is not available for PreComputedGaitCycle. Cannot exclude vertical.")

        if algorithm == MeanSpeedAlgorithm.TOTAL_DISTANCE_OVER_TOTAL_TIME:
            return self._stride_length / self._stride_time
        elif algorithm == MeanSpeedAlgorithm.AVERAGE_INSTANTANEOUS_SPEED:
            raise ValueError(
                "Mean gait speed is only available with MeanSpeedAlgorithm.TOTAL_DISTANCE_OVER_TOTAL_TIME for PreComputedGaitCycle."
            )
        else:
            raise ValueError("Unsupported mean speed algorithm.")

    def stride_length(self, exclude_vertical: bool = False) -> float:
        if exclude_vertical:
            raise ValueError("Vertical component is not available for PreComputedGaitCycle. Cannot exclude vertical.")
        return self._stride_length

    @property
    def stride_time(self) -> float:
        return self._stride_time

    @property
    def stance_time(self) -> float:
        raise ValueError("Stance time is not available for PreComputedGaitCycle.")

    @property
    def swing_time(self) -> float:
        raise ValueError("Swing time is not available for PreComputedGaitCycle.")


class SwayMetrics(Enum):
    AMPLITUDE_AP = auto()
    AMPLITUDE_ML = auto()
    LENGTH = auto()
    VELOCITY = auto()
    CONFIDENCE_ELLIPSE_AREA = auto()


class SwayTrial(AnalysesData):
    def __init__(
        self,
        center_of_mass_data: np.ndarray,
        frame_rate: int,
        starting_index_in_data: int,
    ):
        self._center_of_mass_data = center_of_mass_data
        self._frame_rate = frame_rate
        self._starting_index_in_data = starting_index_in_data

    def indices(self) -> tuple[int, int]:
        return self._starting_index_in_data, self._starting_index_in_data + self._center_of_mass_data.shape[1]

    def length(self, direction: SwayDirection) -> float:
        if direction == SwayDirection.ANTERO_POSTERIOR:
            data_slice = 0
        elif direction == SwayDirection.MEDIO_LATERAL:
            data_slice = 1
        elif direction == SwayDirection.HORIZONTAL_PLANE:
            data_slice = slice(2)
        else:
            raise ValueError("Unsupported sway direction.")
        return np.sum(np.linalg.norm(np.diff(self._center_of_mass_data[data_slice, :], axis=1), axis=0))

    @staticmethod
    def mean_length_from_trials(trials: list[SwayTrial], direction: SwayDirection) -> float:
        if not trials:
            return np.nan
        return np.mean([trial.length(direction=direction) for trial in trials if trial is not None])

    def amplitude(self, direction: SwayDirection) -> float:
        if direction == SwayDirection.ANTERO_POSTERIOR:
            data_slice = 0
        elif direction == SwayDirection.MEDIO_LATERAL:
            data_slice = 1
        elif direction == SwayDirection.HORIZONTAL_PLANE:
            data_slice = slice(2)
        else:
            raise ValueError("Unsupported sway direction.")
        return np.ptp(self._center_of_mass_data[data_slice, :], axis=0)

    @staticmethod
    def mean_amplitude_from_trials(trials: list[SwayTrial], direction: SwayDirection) -> float:
        if not trials:
            return np.nan
        return np.mean([trial.amplitude(direction=direction) for trial in trials if trial is not None])

    def velocity(self, direction: SwayDirection) -> np.ndarray:
        if direction == SwayDirection.ANTERO_POSTERIOR:
            data_slice = 0
        elif direction == SwayDirection.MEDIO_LATERAL:
            data_slice = 1
        elif direction == SwayDirection.HORIZONTAL_PLANE:
            data_slice = slice(2)
        else:
            raise ValueError("Unsupported sway direction.")
        return np.linalg.norm(derivative(self._center_of_mass_data[data_slice, :], frame_rate=self._frame_rate), axis=0)

    def mean_velocity(self, direction: SwayDirection) -> float:
        return np.mean(self.velocity(direction=direction))

    @staticmethod
    def mean_mean_velocity_from_trials(trials: list[SwayTrial], direction: SwayDirection) -> float:
        if not trials:
            return np.nan
        return np.mean([trial.mean_velocity(direction=direction) for trial in trials if trial is not None])

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

    def confidence_ellipse_area(self, confidence_level: float = 0.95) -> float:
        (width, height), _ = self.confidence_ellipse(confidence_level=confidence_level)
        return np.pi * (width / 2) * (height / 2)

    @staticmethod
    def mean_confidence_ellipse_area_from_trials(trials: list[SwayTrial], confidence_level: float = 0.95) -> float:
        if not trials:
            return np.nan
        return np.mean(
            [trial.confidence_ellipse_area(confidence_level=confidence_level) for trial in trials if trial is not None]
        )


class PreComputedSwayTrial(SwayTrial):
    def __init__(
        self,
        antero_posterior_amplitude: float,
        medio_lateral_amplitude: float,
        mean_velocity: float,
        confidence_ellipse_area: float,
        length: float,
    ):
        self._antero_posterior_amplitude = antero_posterior_amplitude
        self._medio_lateral_amplitude = medio_lateral_amplitude
        self._mean_velocity = mean_velocity
        self._confidence_ellipse_area = confidence_ellipse_area
        self._length = length

    def length(self, direction: SwayDirection) -> int:
        if direction != SwayDirection.HORIZONTAL_PLANE:
            raise ValueError("Length is only available for horizontal plane in PreComputedSwayTrial.")
        return self._length

    def amplitude(self, direction: SwayDirection) -> np.ndarray:
        if direction == SwayDirection.ANTERO_POSTERIOR:
            return self._antero_posterior_amplitude
        elif direction == SwayDirection.MEDIO_LATERAL:
            return self._medio_lateral_amplitude
        else:
            raise ValueError("Unsupported sway direction.")

    def velocity(self, direction: SwayDirection) -> np.ndarray:
        raise ValueError("Velocity is not available for PreComputedSwayTrial. Use mean_velocity instead.")

    def mean_velocity(self, direction: SwayDirection) -> np.ndarray:
        if direction != SwayDirection.HORIZONTAL_PLANE:
            raise ValueError("Velocity is only available for horizontal plane in PreComputedSwayTrial.")
        return self._mean_velocity

    def confidence_ellipse(self, confidence_level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError(
            "Confidence ellipse is not available for PreComputedSwayTrial. Use confidence_ellipse_area instead."
        )

    def confidence_ellipse_area(self, confidence_level: float = 0.95) -> float:
        if confidence_level != 0.95:
            raise ValueError("Confidence ellipse is only available for 95% confidence level in PreComputedSwayTrial.")
        return self._confidence_ellipse_area
