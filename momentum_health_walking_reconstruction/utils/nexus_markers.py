from enum import Enum
from typing import Iterable

from biobuddy import C3dData


class NexusMarkers:
    def __init__(self, file: C3dData, expected_marker_names: Iterable[Enum]):
        self._markers = {}
        for m in expected_marker_names:
            is_found = False
            for name in file.marker_names:
                name: str
                if name.endswith(m.value):
                    self._markers[m] = name
                    is_found = True
                    break
            if not is_found:
                raise RuntimeError(f"Marker {m.value} not found in the C3D file.")

    def __getitem__(self, marker: Enum) -> str:
        return self._markers[marker]
