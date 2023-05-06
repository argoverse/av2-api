//! # SE(3)
//!
//! Special Euclidean Group 3.

use std::f32::consts::PI;

use ndarray::{concatenate, s, Array, Array1, Array2, ArrayView, ArrayView2, Axis, Ix2};

use super::so3::{quat_to_yaw, yaw_to_quat};

/// Special Euclidean Group 3 (SE(3)).
/// Rigid transformation parameterized by a rotation and translation in $R^3$.
#[derive(Clone, Debug)]
pub struct SE3 {
    /// (3,3) Orthonormal rotation matrix.
    pub rotation: Array2<f32>,
    /// (3,) Translation vector.
    pub translation: Array1<f32>,
}

impl SE3 {
    /// Get the (4,4) homogeneous transformation matrix associated with the rigid transformation.
    pub fn transform_matrix(&self) -> Array2<f32> {
        let mut transform_matrix = Array2::eye(4);
        transform_matrix
            .slice_mut(s![..3, ..3])
            .assign(&self.rotation);
        transform_matrix
            .slice_mut(s![..3, 3])
            .assign(&self.translation);
        transform_matrix
    }

    /// Transform the point cloud from its reference from to the SE(3) destination.
    pub fn transform_from(&self, point_cloud: &ArrayView2<f32>) -> Array2<f32> {
        point_cloud.dot(&self.rotation.t()) + &self.translation
    }

    /// Invert the SE(3) transformation.
    pub fn inverse(&self) -> SE3 {
        let rotation = self.rotation.t().as_standard_layout().to_owned();
        let translation = rotation.dot(&(-&self.translation));
        Self {
            rotation,
            translation,
        }
    }

    /// Compose (right multiply) an SE(3) with another SE(3).
    pub fn compose(&self, right_se3: &SE3) -> SE3 {
        let chained_transform_matrix = self.transform_matrix().dot(&right_se3.transform_matrix());
        SE3 {
            rotation: chained_transform_matrix
                .slice(s![..3, ..3])
                .as_standard_layout()
                .to_owned(),
            translation: chained_transform_matrix
                .slice(s![..3, 3])
                .as_standard_layout()
                .to_owned(),
        }
    }
}

/// Reflect pose across the x-axis.
/// (N,7) `xyz_qwxyz` represent the translation and orientation of `N` rigid objects.
pub fn reflect_pose_x(xyz_qwxyz: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let xyz_m = xyz_qwxyz.slice(s![.., ..3]);
    let quat_wxyz = xyz_qwxyz.slice(s![.., -4..]);

    let mut reflected_xyz_m = xyz_m.clone().to_owned();
    reflected_xyz_m
        .slice_mut(s![.., 1])
        .par_mapv_inplace(|y| -y);

    let yaw_rad = quat_to_yaw(&quat_wxyz);
    let reflected_yaw_rad = -yaw_rad;
    let reflected_quat_wxyz = yaw_to_quat(&reflected_yaw_rad.view());
    concatenate![Axis(1), reflected_xyz_m, reflected_quat_wxyz]
}

/// Reflect pose across the y-axis.
/// (N,7) `xyz_qwxyz` represent the translation and orientation of `N` rigid objects.
pub fn reflect_pose_y(xyz_qwxyz: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let xyz_m = xyz_qwxyz.slice(s![.., ..3]);
    let quat_wxyz = xyz_qwxyz.slice(s![.., -4..]);

    let mut reflected_xyz_m = xyz_m.clone().to_owned();
    reflected_xyz_m
        .slice_mut(s![.., 0])
        .par_mapv_inplace(|x| -x);

    let yaw_rad = quat_to_yaw(&quat_wxyz);
    let reflected_yaw_rad = PI - yaw_rad;
    let reflected_quat_wxyz = yaw_to_quat(&reflected_yaw_rad.view());
    concatenate![Axis(1), reflected_xyz_m, reflected_quat_wxyz]
}
