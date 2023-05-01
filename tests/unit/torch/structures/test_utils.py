"""Unit test for PyTorch structure utilities."""

import numpy as np
import pandas as pd
import torch
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion
from torch.testing._comparison import assert_close

from av2.torch import QWXYZ_COLUMNS, TRANSLATION_COLUMNS
from av2.torch.structures.utils import SE3_from_frame, tensor_from_frame


def _build_dummy_frame() -> pd.DataFrame:
    """Build a dummy data-frame."""
    return pd.DataFrame(
        {
            "tx_m": [0.0],
            "ty_m": [1.0],
            "tz_m": [0.0],
            "qw": [1.0],
            "qx": [0.0],
            "qy": [0.0],
            "qz": [0.0],
        },
        dtype=np.float32,
    )


def test_tensor_from_frame() -> None:
    """Test converting a data-frame into a tensor."""
    frame = _build_dummy_frame()
    tensor = tensor_from_frame(frame, columns=["qw", "qx", "qy", "qz"])

    tensor_expected = torch.as_tensor(
        [
            [
                frame.loc[0, "qw"],
                frame.loc[0, "qx"],
                frame.loc[0, "qy"],
                frame.loc[0, "qz"],
            ]
        ]
    )
    assert_close(tensor, tensor_expected)


def test_SE3_from_frame() -> None:
    """Test converting a data-frame into an SE(3) object."""
    frame = _build_dummy_frame()

    quat_wxyz_tensor = torch.as_tensor(
        frame[list(QWXYZ_COLUMNS)].to_numpy().astype(np.float32)
    )
    translation = torch.as_tensor(
        frame[list(TRANSLATION_COLUMNS)].to_numpy().astype(np.float32)
    )
    quat_wxyz = Quaternion(quat_wxyz_tensor)
    rotation = So3(quat_wxyz)
    city_SE3_ego_expected = Se3(rotation, translation)
    city_SE3_ego = SE3_from_frame(frame)

    assert_close(city_SE3_ego.translation, city_SE3_ego_expected.translation)
    assert_close(
        city_SE3_ego.rotation.matrix(), city_SE3_ego_expected.rotation.matrix()
    )
