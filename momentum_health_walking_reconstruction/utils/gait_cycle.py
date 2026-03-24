from enum import Enum, auto

import numpy as np


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()


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
