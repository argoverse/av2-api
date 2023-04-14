# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Types used throughout the package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union  # noqa

import numpy as np
import numpy.typing as npt
from upath import UPath

NDArrayNumber = npt.NDArray["np.number[Any]"]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayFloat = npt.NDArray[np.float64]
NDArrayFloat32 = npt.NDArray[np.float32]
NDArrayByte = npt.NDArray[np.uint8]
NDArrayInt = npt.NDArray[np.int64]
NDArrayObject = npt.NDArray[np.object_]

PathType = Union[Path, UPath]
