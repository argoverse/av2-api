use std::path::Path;

use ndarray::{par_azip, s, Array, ArrayView, Ix1, Ix2};
use polars::{
    lazy::dsl::{col, lit},
    prelude::{DataFrame, IntoLazy},
};

use crate::{geometry::utils::cart_to_hom, io::read_feather_eager, se3::SE3, so3::quat_to_mat3};

/// Pinhole camera intrinsics.
#[derive(Clone)]
pub struct Intrinsics {
    /// Horizontal focal length in pixels.
    pub fx_px: f32,
    /// Vertical focal length in pixels.
    pub fy_px: f32,
    /// Horizontal focal center in pixels.
    pub cx_px: f32,
    /// Vertical focal center in pixels.
    pub cy_px: f32,
    /// Width of image in pixels.
    pub width_px: usize,
    /// Height of image in pixels.
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
            height_px,
            width_px,
        }
    }

    /// Camera intrinsic matrix.
    pub fn k(&self) -> Array<f32, Ix2> {
        let mut k = Array::<f32, Ix2>::eye(2);
        k[[0, 0]] = self.fx_px;
        k[[1, 1]] = self.fy_px;
        k[[0, 2]] = self.cx_px;
        k[[1, 2]] = self.cy_px;
        k
    }
}

/// Parameterizes a pinhole camera with zero skew.
#[derive(Clone)]
pub struct PinholeCamera {
    /// Pose of camera in the egovehicle frame (inverse of extrinsics matrix).
    pub ego_se3_cam: SE3,
    /// `Intrinsics` object containing intrinsic parameters and image dimensions.
    pub intrinsics: Intrinsics,
    /// Associated camera name.
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
        let rotation = quat_to_mat3(&quat_wxyz.view());
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

    /// Cull 3d points to camera view frustum.
    ///
    /// Ref: https://en.wikipedia.org/wiki/Hidden-surface_determination#Viewing-frustum_culling
    ///
    /// Given a set of coordinates in the image plane and corresponding points
    /// in the camera coordinate reference frame, determine those points
    /// that have a valid projection into the image. 3d points with valid
    /// projections have x coordinates in the range [0,width_px-1], y-coordinates
    /// in the range [0,height_px-1], and a positive z-coordinate (lying in
    /// front of the camera frustum).
    pub fn cull_to_view_frustum(
        self,
        uv: &ArrayView<f32, Ix2>,
        points_cam: &ArrayView<f32, Ix2>,
    ) -> Array<bool, Ix2> {
        let num_points = uv.shape()[0];
        let mut is_valid_points = Array::<bool, Ix1>::from_vec(vec![false; num_points])
            .into_shape([num_points, 1])
            .unwrap();
        par_azip!((mut is_valid in is_valid_points.outer_iter_mut(), uv_row in uv.outer_iter(), point_cam in points_cam.outer_iter()) {
            let is_valid_x = (0. < uv_row[0]) && (uv_row[0] < (self.width_px() - 1) as f32);
            let is_valid_y = (0. < uv_row[1]) && (uv_row[1] < (self.height_px() - 1) as f32);
            let is_valid_z = point_cam[2] > 0.;
            is_valid[0] = is_valid_x & is_valid_y & is_valid_z;
        });
        is_valid_points
    }

    /// Project a collection of 3d points (provided in the egovehicle frame) to the image plane.
    pub fn project_ego_to_image(
        &self,
        points_ego: Array<f32, Ix2>,
    ) -> (Array<f32, Ix2>, Array<f32, Ix2>, Array<bool, Ix2>) {
        // Convert cartesian to homogeneous coordinates.
        let points_ego_hom = cart_to_hom(points_ego);
        let points_hom_cam = points_ego_hom.dot(&self.extrinsics().permuted_axes([1, 0]));
        let points_cart_cam = points_hom_cam.slice(s![.., ..3]);
        let uv: Array<f32, Ix2> = points_cart_cam.dot(&self.intrinsics.k());

        let uv = &uv / &uv.slice(s![.., 2]);
        let is_valid_points = self
            .clone()
            .cull_to_view_frustum(&uv.view(), &points_cart_cam);
        (uv, points_cart_cam.to_owned(), is_valid_points)
    }
}

fn extract_f32_from_frame(series: &DataFrame, column: &str) -> f32 {
    series
        .column(column)
        .unwrap()
        .get(0)
        .unwrap()
        .try_extract::<f32>()
        .unwrap()
}

fn extract_usize_from_frame(series: &DataFrame, column: &str) -> usize {
    series[column]
        .get(0)
        .unwrap()
        .try_extract::<usize>()
        .unwrap()
}
