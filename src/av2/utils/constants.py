# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Common mathematical and computing system constants."""

import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Final

# Difference between 1.0 and the least value greater than 1.0 that is representable as a float.
# https://docs.python.org/3/library/sys.html#sys.float_info
EPS: Final[float] = sys.float_info.epsilon

# 3.14159 ...
PI: Final[float] = math.pi

# Not a number.
NAN: Final[float] = math.nan

# System home directory.
HOME: Final[Path] = Path.home()

# Max number of logical cores available on the current machine.
MAX_CPUS: Final[int] = mp.cpu_count()
