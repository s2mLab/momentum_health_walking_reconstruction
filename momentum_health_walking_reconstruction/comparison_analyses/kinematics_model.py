from abc import ABC, abstractmethod

import biorbd
import ezc3d
from matplotlib import pyplot as plt
import numpy as np
import struct
import pygltflib
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.signal import correlate

from ..utils.gait_cycle import Side, GaitCycle


class KinematicsModel(ABC):
    @abstractmethod
    def time_vector(self) -> np.ndarray:
        pass

    @abstractmethod
    def frame_count(self) -> int:
        pass

    @abstractmethod
    def frame_rate(self) -> float:
        pass

    @abstractmethod
    def set_resample_ratio(self, ratio: int) -> None:
        pass

    @abstractmethod
    def knee_sagittal_angles(self, side: Side) -> np.ndarray:
        pass

    @abstractmethod
    def center_of_mass(self) -> np.ndarray:
        pass

    @abstractmethod
    def initial_frame_index(self) -> int:
        pass

    @abstractmethod
    def set_initial_frame_index(self, new_frame_index: int):
        pass

    @abstractmethod
    def last_frame_index(self) -> int:
        pass

    @abstractmethod
    def set_last_frame_index(self, new_frame_index: int):
        pass


def perform_align_kinematics(data1: KinematicsModel, data2: KinematicsModel, side: Side) -> None:
    if data1.frame_count() > data2.frame_count():
        data1.set_resample_ratio(data1.frame_rate(resampled=False) // data2.frame_rate())
    elif data2.frame_count() > data1.frame_count():
        data2.set_resample_ratio(data2.frame_rate(resampled=False) // data1.frame_rate())

    data_knee_angles = data1.knee_sagittal_angles(side, resampled=True)
    reference_knee_angles = data2.knee_sagittal_angles(side)

    corr = correlate(data_knee_angles, reference_knee_angles, mode="full")
    lags = np.arange(-len(reference_knee_angles) + 1, len(data_knee_angles))
    best_lag = lags[np.argmax(corr)]

    if best_lag > 0:
        data1.set_initial_frame_index(data1.initial_frame_index() + best_lag)
    elif best_lag < 0:
        data2.set_initial_frame_index(data2.initial_frame_index() - best_lag)

    if data1.frame_count() > data2.frame_count():
        data1.set_last_frame_index(data2.frame_count() - 1)
    elif data2.frame_count() > data1.frame_count():
        data2.set_last_frame_index(data1.frame_count() - 1)


class BiorbdKinematicsModel(KinematicsModel):
    def __init__(self, model: biorbd.Biorbd, c3d_data: ezc3d.c3d, kinematics: np.ndarray):
        self._model = model

        # Load the inhouse model data along with the data used to compute the kinematics
        self._c3d = c3d_data
        self._kinematics: np.ndarray = kinematics

        self._resample_ratio = 1
        self._initial_frame_index = 0
        self._last_frame_index = self._kinematics.shape[1]

    @classmethod
    def from_file(cls, model_path: str, c3d_path: str, kinematics_path: str):
        model = biorbd.Biorbd(model_path)

        # Load the inhouse model data along with the data used to compute the kinematics
        c3d_data = ezc3d.c3d(c3d_path)
        kinematics = np.load(kinematics_path, allow_pickle=True)

        return cls(model=model, c3d_data=c3d_data, kinematics=kinematics)

    def time_vector(self, resampled: bool = True) -> np.ndarray:
        return np.arange(self.frame_count(resampled=resampled)) / self.frame_rate(resampled=resampled)

    def frame_count(self, resampled: bool = True) -> int:
        return self.kinematics(resampled=resampled).shape[1]

    def kinematics(self, resampled: bool = True) -> np.ndarray:
        initial_frame_index = self.initial_frame_index(resampled=False)
        last_frame_index = self.last_frame_index(resampled=False)
        resample_ratio = self.resample_ratio(resampled=resampled)
        return self._kinematics[:, initial_frame_index:last_frame_index:resample_ratio]

    def point_data(self, resampled: bool = True) -> np.ndarray:
        initial_frame_index = self.initial_frame_index(resampled=False)
        last_frame_index = self.last_frame_index(resampled=False)
        resample_ratio = self.resample_ratio(resampled=resampled)
        return self._c3d.data["points"][:3, :, initial_frame_index:last_frame_index:resample_ratio] / 1000.0

    def resample_ratio(self, resampled: bool = True) -> int:
        return self._resample_ratio if resampled else 1

    def set_resample_ratio(self, ratio: int) -> None:
        if ratio < 1:
            raise ValueError("Resample ratio must be a positive integer.")

        if self._c3d.header["points"]["frame_rate"] % ratio != 0:
            raise ValueError("Resample ratio must be a divisor of the original frame rate.")
        self._resample_ratio = int(ratio)

    def initial_frame_index(self, resampled: bool = True) -> int:
        return self._initial_frame_index // self.resample_ratio(resampled=resampled)

    def set_initial_frame_index(self, new_frame_index: int):
        new_frame_index_in_original = new_frame_index * self.resample_ratio(resampled=True)

        if new_frame_index_in_original < 0 or new_frame_index_in_original >= self._kinematics.shape[1]:
            raise ValueError("Offset must be a non-negative integer.")
        self._initial_frame_index = new_frame_index_in_original

    def last_frame_index(self, resampled: bool = True) -> int:
        return self._last_frame_index // self.resample_ratio(resampled=resampled)

    def set_last_frame_index(self, new_frame_index: int):
        initial_frame_index = self.initial_frame_index(resampled=False)
        new_frame_index_in_original = new_frame_index * self.resample_ratio(resampled=True) + initial_frame_index

        if new_frame_index_in_original < 0 or new_frame_index_in_original >= self._kinematics.shape[1]:
            raise ValueError("Offset must be a non-negative integer.")
        self._last_frame_index = new_frame_index_in_original

    def frame_rate(self, resampled: bool = True) -> float:
        return self._c3d.header["points"]["frame_rate"] // self.resample_ratio(resampled=resampled)

    @property
    def point_names(self) -> list[str]:
        return self._c3d.parameters["POINT"]["LABELS"]["value"]

    def knee_sagittal_angles(self, side: Side, resampled: bool = True) -> np.ndarray:
        if side == Side.LEFT:
            prefix = "L"
        elif side == Side.RIGHT:
            prefix = "R"
        else:
            raise ValueError("Invalid side. Must be Side.LEFT or Side.RIGHT.")

        index_knee = self._model.dof_names.index(f"{prefix}Shank_RotX")
        return self.kinematics(resampled=resampled)[index_knee, :]

    def center_of_mass(self, resampled: bool = True) -> np.ndarray:
        q = self.kinematics(resampled=resampled)
        return np.array([self._model.center_of_mass(q[:, index]) for index in range(q.shape[1])]).T

    def center_of_mass_velocity(self, resampled: bool = True) -> np.ndarray:
        return np.gradient(
            self.center_of_mass(resampled=resampled), 1 / self.frame_rate(resampled=resampled), axis=1, edge_order=2
        )

    def gait_cycles(self, side: Side, resampled: bool = True) -> list[GaitCycle]:
        if side == Side.LEFT:
            prefix = "L"
        elif side == Side.RIGHT:
            prefix = "R"
        else:
            raise ValueError("Invalid side. Must be Side.LEFT or Side.RIGHT.")

        return _extract_gait_cycles(
            point_data=self.point_data(resampled=resampled),
            center_of_mass_data=self.center_of_mass(resampled=resampled),
            frame_rate=self.frame_rate(resampled=resampled),
            point_names=self.point_names,
            heel_point_name=f"{prefix}HEE",
            toe_point_name=f"{prefix}TOE5",
        )


class GlbKinematicsModel(KinematicsModel):
    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data

        self._initial_frame_index = 0
        self._last_frame_index = next(iter(data.values())).shape[0]

    def time_vector(self) -> np.ndarray:
        return np.arange(self.frame_count()) / self.frame_rate()

    def frame_count(self) -> int:
        return self._last_frame_index - self._initial_frame_index

    def frame_rate(self) -> float:
        return 30.0

    def set_resample_ratio(self, ratio: int) -> None:
        raise NotImplementedError(
            "Resampling is not supported for GLB data since it is already at a fixed frame rate of 30 FPS."
        )

    def initial_frame_index(self, resampled: bool = True) -> int:
        return self._initial_frame_index

    def set_initial_frame_index(self, new_frame_index: int):
        new_frame_index_in_original = new_frame_index

        if new_frame_index_in_original < 0 or new_frame_index_in_original >= self.frame_count():
            raise ValueError("Offset must be a non-negative integer.")
        self._initial_frame_index = new_frame_index_in_original

    def last_frame_index(self):
        return self._last_frame_index

    def set_last_frame_index(self, new_frame_index: int):
        new_frame_index_in_original = new_frame_index

        if new_frame_index_in_original < 0 or new_frame_index_in_original >= self.frame_count():
            raise ValueError("Offset must be a non-negative integer.")
        self._last_frame_index = new_frame_index_in_original

    def knee_sagittal_angles(self, side: Side) -> np.ndarray:
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

        first = self._initial_frame_index
        last = self._last_frame_index
        hip_to_knee = self._data[knee_name][first:last, :] - self._data[hip_name][first:last, :]
        knee_to_ankle = self._data[ankle_name][first:last, :] - self._data[knee_name][first:last, :]
        hip_to_knee_norm = hip_to_knee / (np.linalg.norm(hip_to_knee, axis=1, keepdims=True) + 1e-6)
        knee_to_ankle_norm = knee_to_ankle / (np.linalg.norm(knee_to_ankle, axis=1, keepdims=True) + 1e-6)
        return np.arccos(np.clip(np.sum(hip_to_knee_norm * knee_to_ankle_norm, axis=1), -1, 1))

    def center_of_mass(self) -> np.ndarray:
        raise NotImplementedError("Center of mass data is not available in the GLB file.")

    @classmethod
    def from_file(cls, glb_path: str) -> "GlbKinematicsModel":
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
        return cls(data=glb_pos)


def _extract_gait_cycles(
    point_data: np.ndarray,
    center_of_mass_data: np.ndarray,
    frame_rate: int,
    point_names: list[str],
    heel_point_name: str,
    toe_point_name: str,
    show_plot: bool = False,
) -> list[GaitCycle]:
    heel_index = _find_marker_index(point_names, heel_point_name)
    toe_index = _find_marker_index(point_names, toe_point_name)

    cycle_indices = _get_gait_cycle_indices(
        expect_cycle_duration=1.0,
        frame_rate=frame_rate,
        heel_data=point_data[:3, heel_index, :].T,
        toe_data=point_data[:3, toe_index, :].T,
        show_plot=show_plot,
    )
    cycles: list[GaitCycle] = []
    for indices in cycle_indices:
        cycles.append(
            GaitCycle.from_data(
                toe_off_index=indices[1] - indices[0],
                heel_data=point_data[:3, heel_index, indices[0] : indices[2]],
                center_of_mass_data=center_of_mass_data[:3, indices[0] : indices[2]],
                frame_rate=frame_rate,
                starting_index_in_data=indices[0],
            )
        )
    return cycles


def _get_gait_cycle_indices(
    expect_cycle_duration: float,
    frame_rate: int,
    heel_data: np.ndarray,
    toe_data: np.ndarray,
    maximum_peak_threshold: float = 0.75,
    minimum_velocity_threshold: float = 0.5,
    minimum_zeros_frame: int = 10,
    show_plot: bool = False,
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
        plt.plot(heel_velocity, label="LHeel")
        plt.plot(ltoe_velocity, label="LToe")
        plt.plot(mid_swing_peaks, heel_velocity[mid_swing_peaks], "o", label="Peaks")
        plt.plot(heel_strikes, np.zeros_like(heel_velocity[heel_strikes]), "x", label="Heel Strikes")
        plt.plot(toe_offs, np.zeros_like(ltoe_velocity[toe_offs]), "s", label="Toe Offs")
        # Plot the cycles as shaded areas
        for start, toe_off, end in gait_cycles:
            plt.axvspan(start, end, color="gray", alpha=0.3)
        plt.legend()
        plt.title("LHeel trajectory")
        plt.xlabel("Frame")
        plt.ylabel("Velocity (m/s)")
        plt.show()

    return gait_cycles


def _find_marker_index(marker_names: list[str], suffix: str) -> int:
    """Find the index of a marker in the marker names list based on a suffix."""
    for i, name in enumerate(marker_names):
        if name.endswith(suffix):
            return i
    raise ValueError(f"Marker with suffix '{suffix}' not found in marker names.")


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
