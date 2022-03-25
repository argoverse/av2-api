# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on a helper class for generating bird's eye view (BEV) images."""

import numpy as np

from av2.structs.bev_params import BEVParams
from av2.utils.typing import NDArrayFloat


def test_bev_params() -> None:
    """Ensure that for a 1000x1000 image, we do not try to index into index=1000."""
    params = BEVParams(range_m=50, res_meters_per_px=0.1)
    query: NDArrayFloat = np.array([3.76367188, 49.96875])

    # ymax_m should be 49.9
    assert params.ymax_m < query[1]