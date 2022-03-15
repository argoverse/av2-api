# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""SE3 class for point cloud rotation and translation."""
from __future__ import annotations

import numpy as np

from av2.utils.typing import NDArrayFloat


class SE3:
    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation: NDArrayFloat, translation: NDArrayFloat) -> None:
        """Initialize an SE3 instance with its rotation and translation matrices.

        Args:
            rotation: Array of shape (3, 3)
            translation: Array of shape (3,)

        Raises:
            ValueError: If rotation or translation matrix are of incorrect shape.
        """
        if not rotation.shape == (3, 3):
            raise ValueError("Rotation matrix is of valid shape, must be (3, 3)")
        if not translation.shape == (3,):
            raise ValueError("Translation matrix is of invalid shape, must be (3,)")
        self.rotation = rotation
        self.translation = translation

        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = self.rotation
        self.transform_matrix[:3, 3] = self.translation

    def transform_from(self, point_cloud: NDArrayFloat) -> NDArrayFloat:
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`

        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return point_cloud @ self.rotation.T + self.translation

    def transform_point_cloud(self, point_cloud: NDArrayFloat) -> NDArrayFloat:
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`

        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return self.transform_from(point_cloud)

    def inverse(self) -> SE3:
        """Return the inverse of the current SE3 transformation.

        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.

        Returns:
            instance of SE3 class, representing inverse of SE3 transformation target_SE3_src.
        """
        return SE3(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def compose(self, right_SE3: SE3) -> SE3:
        """Compose (right multiply) this class' transformation matrix T with another SE(3) instance.

        Algebraic representation: chained_se3 = T * right_SE3

        Args:
            right_SE3: Another instance of SE3 class.

        Returns:
            New instance of SE3 class.
        """
        chained_transform_matrix: NDArrayFloat = self.transform_matrix @ right_SE3.transform_matrix
        chained_SE3 = SE3(
            rotation=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_SE3
