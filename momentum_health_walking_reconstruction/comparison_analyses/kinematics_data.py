from abc import ABC, abstractmethod
from enum import Enum, auto

import biorbd
import ezc3d
from matplotlib import pyplot as plt
import numpy as np
import struct
import pygltflib
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from scipy.signal import correlate

from ..utils.math import derivative


class TrialType(Enum):
    GAIT = auto()
    SWAY = auto()


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()


class Joint(Enum):
    # HIP = auto()
    KNEE = auto()
    # ANKLE = auto()


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

    @property
    def trial_type(self) -> TrialType:
        return self._trial_type

    def time_vector(self, resampled: bool = True) -> np.ndarray:
        full_time_vector = np.arange(self.frame_count(resampled=False)) / self.frame_rate(resampled=False)
        return full_time_vector[:: self.resample_ratio(resampled=resampled)]

    def frame_count(self, resampled: bool = True) -> int:
        return (
            self.last_frame_index(resampled=False) - self.initial_frame_index(resampled=False)
        ) // self.resample_ratio(resampled=resampled)

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

    def set_last_frame_index(self, new_frame_index: int):
        initial_frame_index = self.initial_frame_index(resampled=False)
        new_frame_index_in_original = new_frame_index * self.resample_ratio(resampled=True) + initial_frame_index

        if new_frame_index_in_original < 0 or new_frame_index_in_original >= self._original_last_frame_index:
            raise ValueError("Offset must be a non-negative integer.")
        self._last_frame_index = new_frame_index_in_original

    def _data_slice(self, resampled: bool = True) -> slice:
        initial_frame_index = self.initial_frame_index(resampled=False)
        last_frame_index = self.last_frame_index(resampled=False)
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

    @staticmethod
    def perform_align_kinematics_data(
        data1: KinematicsData, data2: KinematicsData, side: Side, show_plot: bool = False
    ) -> None:
        if data1.frame_count() > data2.frame_count():
            data1.set_resample_ratio(data1.frame_rate(resampled=False) // data2.frame_rate())
        elif data2.frame_count() > data1.frame_count():
            data2.set_resample_ratio(data2.frame_rate(resampled=False) // data1.frame_rate())

        data_knee_angles = data1.angles(joint=Joint.KNEE, side=side, resampled=True)
        reference_knee_angles = data2.angles(joint=Joint.KNEE, side=side, resampled=True)

        corr = correlate(data_knee_angles, reference_knee_angles, mode="full")
        lags = np.arange(-len(reference_knee_angles) + 1, len(data_knee_angles))
        best_lag = lags[np.argmax(corr)]

        if best_lag > 0:
            data1.set_initial_frame_index(data1.initial_frame_index() + best_lag)
        elif best_lag < 0:
            data2.set_initial_frame_index(data2.initial_frame_index() - best_lag)

        data1_remaining_frames = data1.original_last_frame_index() - data1.initial_frame_index()
        data2_remaining_frames = data2.original_last_frame_index() - data2.initial_frame_index()
        if data1_remaining_frames > data2_remaining_frames:
            data1.set_last_frame_index(data2_remaining_frames)
        elif data2_remaining_frames > data1_remaining_frames:
            data2.set_last_frame_index(data1_remaining_frames)

        if show_plot:
            plt.figure()
            plt.plot(
                data1.time_vector(resampled=True),
                data1.angles(side=side, joint=Joint.KNEE, resampled=True),
                "r--",
                label="Data1 (resampled)",
            )
            plt.plot(
                data1.time_vector(resampled=False),
                data1.angles(side=side, joint=Joint.KNEE, resampled=False),
                "r-",
                label="Data1 (original)",
            )
            plt.plot(
                data2.time_vector(resampled=True),
                data2.angles(side=side, joint=Joint.KNEE, resampled=True),
                "b--",
                label="Data2 (resampled)",
            )
            plt.plot(
                data2.time_vector(resampled=False),
                data2.angles(side=side, joint=Joint.KNEE, resampled=False),
                "b-",
                label="Data2 (original)",
            )
            plt.legend()
            plt.title("Knee trajectory")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (degrees)")
            plt.show()


class BiorbdKinematicsData(KinematicsData):
    def __init__(self, model: biorbd.Biorbd, c3d_data: ezc3d.c3d, kinematics: np.ndarray, trial_type: TrialType):
        self._model = model

        # Load the inhouse model data along with the data used to compute the kinematics
        self._c3d = c3d_data
        self._kinematics: np.ndarray = kinematics

        super().__init__(
            original_frame_rate=self._c3d.header["points"]["frame_rate"],
            last_frame_index=kinematics.shape[1],
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
        else:
            raise ValueError("Invalid side. Must be Side.LEFT or Side.RIGHT.")

        if joint == Joint.KNEE:
            joint_prefix = "Shank_RotX"
        else:
            raise ValueError("Unsupported joint. Only KNEE is currently supported.")

        joint_index = self._model.dof_names.index(f"{side_prefix}{joint_prefix}")
        return self._get_kinematics(resampled=resampled)[joint_index, :]

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
    def from_file(cls, model_path: str, c3d_path: str, kinematics_path: str, trial_type: TrialType):
        model = biorbd.Biorbd(model_path)

        # Load the inhouse model data along with the data used to compute the kinematics
        c3d_data = ezc3d.c3d(c3d_path)
        kinematics = np.load(kinematics_path, allow_pickle=True)

        return cls(model=model, c3d_data=c3d_data, kinematics=kinematics, trial_type=trial_type)

    @staticmethod
    def _find_marker_index(marker_names: list[str], suffix: str) -> int:
        """Find the index of a marker in the marker names list based on a suffix."""
        for i, name in enumerate(marker_names):
            if name.endswith(suffix):
                return i
        raise ValueError(f"Marker with suffix '{suffix}' not found in marker names.")


class GlbKinematicsData(KinematicsData):
    def __init__(self, data: dict[str, np.ndarray], trial_type: TrialType):
        self._data = data
        super().__init__(
            original_frame_rate=30.0,
            last_frame_index=next(iter(data.values())).shape[0],
            trial_type=trial_type,
        )

    def set_resample_ratio(self, ratio: int) -> None:
        raise NotImplementedError(
            "Resampling is not supported for GLB data since it is already at a fixed frame rate of 30 FPS."
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
            raise NotImplementedError("Center of mass data is not available in the GLB file.")
        else:
            raise ValueError("Unsupported point. Only CENTER_OF_MASS is currently supported.")

    @classmethod
    def from_file(cls, glb_path: str, trial_type: TrialType) -> "GlbKinematicsData":
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
