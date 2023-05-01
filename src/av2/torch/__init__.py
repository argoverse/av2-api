"""Argoverse 2 PyTorch API."""

from typing import Final

LIDAR_COLUMNS: Final = ("x", "y", "z", "intensity")
QWXYZ_COLUMNS: Final = ("qw", "qx", "qy", "qz")
TRANSLATION_COLUMNS: Final = ("tx_m", "ty_m", "tz_m")
XYZLWH_QWXYZ_COLUMNS: Final = (
    TRANSLATION_COLUMNS + ("length_m", "width_m", "height_m") + QWXYZ_COLUMNS
)
