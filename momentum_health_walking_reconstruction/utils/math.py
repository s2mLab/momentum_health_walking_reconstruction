import numpy as np


def nanunwrap(data: np.ndarray) -> np.ndarray:
    """
    Unwrap the data while ignoring NaN values at the beginning of the trial. The unwrapping assumes 1d data

    Parameters
    ----------
    data : np.ndarray
        The input data to unwrap, which may contain NaN values.

    Returns
    -------
    np.ndarray
        The unwrapped data, with NaN values preserved in their original positions.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    # Find the index of the first non-NaN value
    first_valid_index = np.where(~np.isnan(data))[0]
    if len(first_valid_index) == 0:
        # If all values are NaN, return the original data
        return data
    first_valid_index = first_valid_index[0]
    # Unwrap the data starting from the first valid index
    unwrapped_data = np.copy(data)
    unwrapped_data[first_valid_index:] = np.unwrap(data[first_valid_index:])

    return unwrapped_data


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
