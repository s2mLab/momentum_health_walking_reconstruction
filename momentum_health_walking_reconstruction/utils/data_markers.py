from pathlib import Path
from typing import Iterable

import biobuddy
import ezc3d
import numpy as np


class BiobuddyData(biobuddy.Data):
    def __init__(self, data: dict[str, np.ndarray]):
        self.values = data


class DataMarkers:
    def __init__(self, marker_names: Iterable[str], markers: np.ndarray):
        self._marker_names = list(marker_names)
        self._markers = markers

    def __getitem__(self, marker: str) -> str:
        marker_index = self._marker_names.index(marker)
        return self._markers[:, marker_index, :]

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

    def to_biobuddy_data(self) -> BiobuddyData:
        data_dict: dict[str, np.ndarray] = {}
        for i, name in enumerate(self._marker_names):
            data_dict[name] = self._markers[:, i, :]
        return BiobuddyData(data_dict)

    @classmethod
    def from_c3d(cls, file_path: Path) -> "DataMarkers":
        c3d = ezc3d.c3d(file_path.as_posix())

        marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
        markers = c3d["data"]["points"]

        instance = cls(marker_names=marker_names, markers=markers)
        return instance
