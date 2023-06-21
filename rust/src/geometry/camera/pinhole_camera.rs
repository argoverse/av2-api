use std::{ops::DivAssign, path::Path};

use ndarray::{par_azip, s, Array, ArrayView, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use polars::{
    lazy::dsl::{col, lit},
    prelude::{DataFrame, IntoLazy},
};
use pyo3::prelude::*;

use crate::{
    geometry::se3::SE3, geometry::so3::_quat_to_mat3, geometry::utils::cart_to_hom,
    io::read_feather_eager,
};

/// Pinhole camera intrinsics.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Intrinsics {
    /// Horizontal focal length in pixels.
    #[pyo3(get, set)]
    pub fx_px: f32,
    /// Vertical focal length in pixels.
    #[pyo3(get, set)]
    pub fy_px: f32,
    /// Horizontal focal center in pixels.
    #[pyo3(get, set)]
    pub cx_px: f32,
    /// Vertical focal center in pixels.
    #[pyo3(get, set)]
    pub cy_px: f32,
    /// Width of image in pixels.
    #[pyo3(get, set)]
    pub width_px: usize,
    /// Height of image in pixels.
    #[pyo3(get, set)]
    pub height_px: usize,
}

impl Intrinsics {
    /// Construct a new `Intrinsics` instance.
    pub fn new(
        &self,
        fx_px: f32,
        fy_px: f32,
        cx_px: f32,
        cy_px: f32,
        width_px: usize,
        height_px: usize,
    ) -> Self {
        Self {
            fx_px,
            fy_px,
            cx_px,
            cy_px,
            width_px,
            height_px,
        }
    }

    /// Camera intrinsic matrix.
    pub fn k(&self) -> Array<f32, Ix2> {
        let mut k = Array::<f32, Ix2>::eye(3);
        k[[0, 0]] = self.fx_px;
        k[[1, 1]] = self.fy_px;
        k[[0, 2]] = self.cx_px;
        k[[1, 2]] = self.cy_px;
        k
    }
}

#[pymethods]
impl Intrinsics {
    /// Camera intrinsic matrix (python).
    #[pyo3(name = "k")]
    pub fn py_k<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Ix2> {
        self.k().into_pyarray(py)
    }
}

/// Parameterizes a pinhole camera with zero skew.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PinholeCamera {
    /// Pose of camera in the egovehicle frame (inverse of extrinsics matrix).
    #[pyo3(get, set)]
    pub ego_se3_cam: SE3,
    /// `Intrinsics` object containing intrinsic parameters and image dimensions.
    #[pyo3(get, set)]
    pub intrinsics: Intrinsics,
    /// Associated camera name.
    #[pyo3(get, set)]
    pub camera_name: String,
}

impl PinholeCamera {
    /// Return the width of the image in pixels.
    pub fn width_px(&self) -> usize {
        self.intrinsics.width_px
    }

    /// Return the height of the image in pixels."""
    pub fn height_px(&self) -> usize {
        self.intrinsics.height_px
    }

    /// Return the camera extrinsics.
    pub fn extrinsics(&self) -> Array<f32, Ix2> {
        self.ego_se3_cam.inverse().transform_matrix()
    }

    /// Camera projection matrix.
    pub fn p(&self) -> Array<f32, Ix2> {
        let mut p = Array::<f32, Ix2>::zeros([4, 4]);
        p.slice_mut(s![..3, ..3]).assign(&self.intrinsics.k());
        p
    }

    /// Model matrix.
    pub fn m(&self) -> Array<f32, Ix2> {
        self.extrinsics()
    }

    /// Create a pinhole camera model from a feather file.
    pub fn from_feather(log_dir: &Path, camera_name: &str) -> PinholeCamera {
        let intrinsics_path = log_dir.join("calibration/intrinsics.feather");
        let intrinsics = read_feather_eager(&intrinsics_path, false)
            .lazy()
            .filter(col("sensor_name").eq(lit(camera_name)))
            .collect()
            .unwrap();

        let intrinsics = Intrinsics {
            fx_px: extract_f32_from_frame(&intrinsics, "fx_px"),
            fy_px: extract_f32_from_frame(&intrinsics, "fy_px"),
            cx_px: extract_f32_from_frame(&intrinsics, "cx_px"),
            cy_px: extract_f32_from_frame(&intrinsics, "cy_px"),
            width_px: extract_usize_from_frame(&intrinsics, "width_px"),
            height_px: extract_usize_from_frame(&intrinsics, "height_px"),
        };

        let extrinsics_path = log_dir.join("calibration/egovehicle_SE3_sensor.feather");
        let extrinsics = read_feather_eager(&extrinsics_path, false)
            .lazy()
            .filter(col("sensor_name").eq(lit(camera_name)))
            .collect()
            .unwrap();

        let qw = extract_f32_from_frame(&extrinsics, "qw");
        let qx = extract_f32_from_frame(&extrinsics, "qx");
        let qy = extract_f32_from_frame(&extrinsics, "qy");
        let qz = extract_f32_from_frame(&extrinsics, "qz");
        let tx_m = extract_f32_from_frame(&extrinsics, "tx_m");
        let ty_m = extract_f32_from_frame(&extrinsics, "ty_m");
        let tz_m = extract_f32_from_frame(&extrinsics, "tz_m");

        let quat_wxyz = Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz]);
        let rotation = _quat_to_mat3(&quat_wxyz.view());
        let translation = Array::<f32, Ix1>::from_vec(vec![tx_m, ty_m, tz_m]);
        let ego_se3_cam = SE3 {
            rotation,
            translation,
        };

        let camera_name_string = camera_name.to_string();
        Self {
            ego_se3_cam,
            intrinsics,
            camera_name: camera_name_string,
        }
    }

    /// Cull 3D points to camera view frustum.
    ///
    /// Ref: https://en.wikipedia.org/wiki/Hidden-surface_determination#Viewing-frustum_culling
    ///
    /// Given a set of coordinates in the image plane and corresponding points
    /// in the camera coordinate reference frame, determine those points
    /// that have a valid projection into the image. 3D points with valid
    /// projections have x coordinates in the range [0,width_px), y-coordinates
    /// in the range [0,height_px), and a positive z-coordinate (lying in
    /// front of the camera frustum).
    pub fn cull_to_view_frustum(
        self,
        uv: &ArrayView<f32, Ix2>,
        points_camera: &ArrayView<f32, Ix2>,
    ) -> Array<bool, Ix2> {
        let num_points = uv.shape()[0];
        let mut is_within_frustum = Array::<bool, Ix1>::from_vec(vec![false; num_points])
            .into_shape([num_points, 1])
            .unwrap();
        par_azip!((mut is_within_frustum_i in is_within_frustum.outer_iter_mut(), uv_row in uv.outer_iter(), point_cam in points_camera.outer_iter()) {
            let is_within_frustum_x = (uv_row[0] >= 0.) && (uv_row[0] < self.width_px() as f32);
            let is_within_frustum_y = (uv_row[1] >= 0.) && (uv_row[1] < self.height_px() as f32);
            let is_within_frustum_z = point_cam[2] > 0.;
            is_within_frustum_i[0] = is_within_frustum_x & is_within_frustum_y & is_within_frustum_z;
        });
        is_within_frustum
    }

    /// Project a collection of 3D points (provided in the egovehicle frame) to the image plane.
    pub fn project_ego_to_image(
        &self,
        points_ego: ArrayView<f32, Ix2>,
    ) -> (Array<f32, Ix2>, Array<f32, Ix2>, Array<bool, Ix2>) {
        // Convert cartesian to homogeneous coordinates.
        let points_hom_ego = cart_to_hom(points_ego);
        let points_hom_cam = points_hom_ego.dot(&self.extrinsics().t());

        let mut uvz = points_hom_cam
            .slice(s![.., ..3])
            .dot(&self.intrinsics.k().t());
        let z = uvz.slice(s![.., 2..3]).clone().to_owned();
        uvz.slice_mut(s![.., ..2]).div_assign(&z);
        let is_valid_points = self
            .clone()
            .cull_to_view_frustum(&uvz.view(), &points_hom_cam.view());
        (uvz.to_owned(), points_hom_cam.to_owned(), is_valid_points)
    }

    /// Project a collection of 3D points (provided in the egovehicle frame) to the image plane.
    pub fn project_ego_to_image_motion_compensated(
        &self,
        points_ego: ArrayView<f32, Ix2>,
        city_se3_ego_camera_t: SE3,
        city_se3_ego_lidar_t: SE3,
    ) -> (Array<f32, Ix2>, Array<f32, Ix2>, Array<bool, Ix2>) {
        // Convert cartesian to homogeneous coordinates.
        let ego_cam_t_se3_ego_lidar_t = city_se3_ego_camera_t
            .inverse()
            .compose(&city_se3_ego_lidar_t);

        let points_ego = ego_cam_t_se3_ego_lidar_t.transform_from(&points_ego.view());
        self.project_ego_to_image(points_ego.view())
    }
}

#[pymethods]
impl PinholeCamera {
    /// Camera projection matrix.
    #[pyo3(name = "p")]
    pub fn py_p<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Ix2> {
        self.p().into_pyarray(py)
    }

    /// Camera projection matrix.
    #[pyo3(name = "mvp")]
    pub fn py_mvp<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Ix2> {
        self.p().dot(&self.m()).into_pyarray(py)
    }

    /// Project a collection of 3D points (provided in the egovehicle frame) to the image plane.
    #[allow(clippy::type_complexity)]
    #[pyo3(name = "project_ego_to_image")]
    pub fn py_project_ego_to_image<'py>(
        &self,
        py: Python<'py>,
        points_ego: PyReadonlyArray<f32, Ix2>,
    ) -> (
        &'py PyArray<f32, Ix2>,
        &'py PyArray<f32, Ix2>,
        &'py PyArray<bool, Ix2>,
    ) {
        let (uvz, points_hom_cam, is_valid_points) =
            self.project_ego_to_image(points_ego.as_array());
        (
            uvz.into_pyarray(py),
            points_hom_cam.into_pyarray(py),
            is_valid_points.into_pyarray(py),
        )
    }
}

fn extract_f32_from_frame(series: &DataFrame, column_name: &str) -> f32 {
    series[column_name]
        .get(0)
        .unwrap_or_else(|_| panic!("{column_name} does not exist."))
        .try_extract::<f32>()
        .unwrap()
}

fn extract_usize_from_frame(series: &DataFrame, column_name: &str) -> usize {
    series[column_name]
        .get(0)
        .unwrap_or_else(|_| panic!("{column_name} does not exist."))
        .try_extract::<usize>()
        .unwrap()
}
