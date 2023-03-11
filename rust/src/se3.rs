use ndarray::{s, Array1, Array2, ArrayView2};

pub struct SE3 {
    pub rotation: Array2<f32>,
    pub translation: Array1<f32>,
}

impl SE3 {
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
    pub fn transform_from(&self, point_cloud: &ArrayView2<f32>) -> Array2<f32> {
        point_cloud.dot(&self.rotation.t()) + &self.translation
    }

    pub fn inverse(&self) -> SE3 {
        let rotation = self.rotation.t().as_standard_layout().to_owned();
        let translation = rotation.dot(&(-&self.translation));
        SE3 {
            rotation,
            translation,
        }
    }

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
