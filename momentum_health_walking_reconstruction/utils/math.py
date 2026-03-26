import numpy as np


def derivative(data: np.ndarray, frame_rate: int) -> np.ndarray:
    return np.gradient(data, 1 / frame_rate, axis=1, edge_order=2)


def find_first_below_threshold(
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
        if current_index < 0 or current_index >= len(velocity_data):
            return -1

        current_velocity = velocity_data[current_index]
        if current_velocity > threshold:
            buffer = 0
        else:
            buffer += 1
            if buffer > minimum_frame_count:
                return current_index + (-1 * direction * buffer)
        offset += 1
