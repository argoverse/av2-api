# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Constants used for unit tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_root_dir() -> Path:
    """Return the absolute path to the root directory for test data."""
    return Path(__file__).resolve().parent / "test_data"
