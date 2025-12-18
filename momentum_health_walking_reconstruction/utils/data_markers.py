from pathlib import Path
from typing import Iterable

import biobuddy
import biorbd
import ezc3d
import numpy as np


class BiobuddyData(biobuddy.Data):
    def __init__(self, data: dict[str, np.ndarray]):
        self.values = data


class DataMarkers:
    def __init__(self, marker_names: Iterable[str], markers: np.ndarray):
        self._marker_names = list(marker_names)
        self._markers = markers

        if len(self._markers.shape) == 2:
            self._markers = self._markers[:, :, None]
        if len(self._markers.shape) != 3:
            raise ValueError(f"Markers array must be 2D or 3D, got shape {self._markers.shape}.")

        if self._markers.shape[1] != len(self._marker_names):
            raise ValueError(
                f"Number of marker names ({len(self._marker_names)}) does not match number of markers ({self._markers.shape[1]})."
            )
        if self._markers.shape[0] != 3 and self._markers.shape[0] != 4:
            raise ValueError(
                f"Markers array first dimension must be 3 (XYZ) or 4 (XYZW), got {self._markers.shape[0]}."
            )

    @property
    def marker_count(self) -> int:
        return len(self._marker_names)

    def __len__(self) -> int:
        return self._markers.shape[2]

    def __getitem__(self, marker: str) -> str:
        marker_index = self._marker_names.index(marker)
        return self._markers[:, marker_index, :]

    @property
    def marker_names(self) -> list[str]:
        return self._marker_names

    def filter(self, expected_marker_names: Iterable[str], rename_markers: bool = True) -> "DataMarkers":
        new_data: dict[str, np.ndarray] = {}
        for m in expected_marker_names:
            is_found = False
            for name in self._marker_names:
                if name.endswith(m):
                    if is_found:
                        raise RuntimeError(f"Marker {m} found multiple times in the C3D file.")
                    new_data[m if not rename_markers else name] = self[name][:, None, :]
                    is_found = True
            if not is_found:
                raise RuntimeError(f"Marker {m} not found in the C3D file.")

        return DataMarkers(marker_names=new_data.keys(), markers=np.concatenate(list(new_data.values()), axis=1))

    def to_biobuddy(self) -> BiobuddyData:
        data_dict: dict[str, np.ndarray] = {}
        for i, name in enumerate(self._marker_names):
            data_dict[name] = self._markers[:, i, :]
        return BiobuddyData(data_dict)

    def to_biorbd(self) -> list[np.ndarray]:
        data = []
        for i in range(len(self)):
            data.append(self._markers[:3, :, i])
        return data

    def to_bioviz(self) -> np.ndarray:
        return self._markers[:3, :, :]

    def to_numpy(self) -> np.ndarray:
        return self._markers

    @classmethod
    def from_c3d(cls, file_path: Path) -> "DataMarkers":
        c3d = ezc3d.c3d(file_path.as_posix())

        marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
        markers = c3d["data"]["points"]
        markers[:3, :, :] /= 1000.0  # Convert from mm to m

        instance = cls(marker_names=marker_names, markers=markers)
        return instance
