# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Types used throughout the package."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # noqa
from typing import Union

import numpy as np
import numpy.typing as npt
from upath import UPath

NDArrayNumber = npt.NDArray[Union[np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64]]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayFloat = npt.NDArray[Union[np.float16, np.float32, np.float64]]
NDArrayFloat32 = npt.NDArray[np.float32]
NDArrayFloat64 = npt.NDArray[np.float64]
NDArrayByte = npt.NDArray[np.uint8]
NDArrayInt = npt.NDArray[np.int64]
NDArrayObject = npt.NDArray[np.object_]

PathType = Union[Path, UPath]
