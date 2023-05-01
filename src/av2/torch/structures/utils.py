"""PyTorch structure utilities."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion

from .. import QWXYZ_COLUMNS, TRANSLATION_COLUMNS


def tensor_from_frame(frame: pd.DataFrame, columns: List[str]) -> torch.Tensor:
    """Build lidar `torch` tensor from `pandas` dataframe.

    Notation:
        N: Number of rows.
        K: Number of columns.

    Args:
        frame: (N,K) Pandas DataFrame containing N rows with K columns.
        columns: List of DataFrame columns.

    Returns:
        (N,K) tensor containing the frame data.
    """
    frame_npy = frame.loc[:, columns].to_numpy().astype(np.float32)
    return torch.as_tensor(frame_npy)


def SE3_from_frame(frame: pd.DataFrame) -> Se3:
    """Build SE(3) object from `pandas` DataFrame.

    Notation:
        N: Number of rigid transformations.

    Args:
        frame: (N,4) Pandas DataFrame containing quaternion coefficients.

    Returns:
        SE(3) object representing a (N,4,4) tensor of homogeneous transformations.
    """
    quaternion_npy = frame.loc[0, list(QWXYZ_COLUMNS)].to_numpy().astype(float)
    quat_wxyz = Quaternion(torch.as_tensor(quaternion_npy, dtype=torch.float32)[None])
    rotation = So3(quat_wxyz)

    translation_npy = (
        frame.loc[0, list(TRANSLATION_COLUMNS)].to_numpy().astype(np.float32)
    )
    translation = torch.as_tensor(translation_npy, dtype=torch.float32)[None]
    dst_SE3_src = Se3(rotation, translation)
    dst_SE3_src.rotation._q.requires_grad_(False)
    dst_SE3_src.translation.requires_grad_(False)
    return dst_SE3_src
