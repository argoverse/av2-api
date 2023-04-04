//! # SE(3)
//!
//! Special Euclidean Group 3.

use glam::{Mat3, Quat, Vec3};
use ndarray::{s, Array1, Array2, ArrayView2};

/// Special Euclidean Group 3 (SE(3)).
/// Rigid transformation parameterized by a rotation and translation in $R^3$.
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
        SE3 {
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

pub fn interpolate_pose(
    q1: Quat,
    q2: Quat,
    t_0: Vec3,
    t_1: Vec3,
    timestamp_ns_0: usize,
    timestamp_ns_1: usize,
    query_timestamp: usize,
) -> (Mat3, Vec3) {
    if query_timestamp < timestamp_ns_0 || query_timestamp > timestamp_ns_1 {
        panic!("Query timestamp must be within the interval [t0,t1].")
    }

    let s = (query_timestamp - timestamp_ns_0) / (timestamp_ns_1 - timestamp_ns_0);
    let slerp = q1.slerp(q2, s as f32);

    // Interpolate the rotations at the given time:
    let r_interp = Mat3::from_quat(slerp);
    let t_interp = t_0.lerp(t_1, s as f32);
    (r_interp, t_interp)
}
