# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Types used throughout the package."""

from __future__ import annotations

from typing import Any  # noqa

import numpy as np
import numpy.typing as npt
from typing import Union
from pathlib import Path
from upath import UPath

NDArrayNumber = npt.NDArray["np.number[Any]"]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayFloat = npt.NDArray[np.float64]
NDArrayByte = npt.NDArray[np.uint8]
NDArrayInt = npt.NDArray[np.int64]
NDArrayObject = npt.NDArray[np.object_]

PathType = Union[Path, UPath]
