"""
通用常量、工具函数、数据结构
"""
import numpy as np
from pathlib import Path

C_LIGHT_NS = 299792458.0 / 1e9
COORD_FILE = str(Path(__file__).resolve().parents[1] / "_gp65_rtksort.txt")


def load_du_coords(file_path):
    """Load DU IDs and 3D coordinates, filtering comments and invalid lines"""
    du_coords = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                du_id = int(parts[0])
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                du_coords[du_id] = coords
            except Exception:
                continue
    return du_coords


class IncrementalYamlMapWriter:
    """Write one top-level YAML mapping entry at a time."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._handle = open(file_path, "w", encoding="utf-8")
        self._has_entries = False

    def write_entry(self, key: int, payload):
        import yaml
        yaml.dump({str(key): payload}, self._handle, sort_keys=False, default_flow_style=False)
        self._has_entries = True

    def close(self):
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
