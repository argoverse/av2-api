# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Constants used for unit tests."""

from pathlib import Path

import pytest
import numpy as _np

# Restore deprecated numpy scalar aliases removed in newer numpy versions
# Some third-party libs (e.g., trackeval) still reference `np.float`, `np.int`,
# etc. Add safe fallbacks so tests run with modern numpy.
_np_aliases = {
    "float": float,
    "int": int,
    "bool": bool,
    "complex": complex,
    "object": object,
    "str": str,
}
for _name, _typ in _np_aliases.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _typ)


@pytest.fixture(scope="session")
def test_data_root_dir() -> Path:
    """Return the absolute path to the root directory for test data."""
    return Path(__file__).resolve().parent / "test_data"
