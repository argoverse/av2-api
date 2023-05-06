//! # SE(3)
//!
//! Special Euclidean Group 3.

use ndarray::ArrayView2;
use ndarray::{s, Array1, Array2};

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
