use std::path::Path;

use ndarray::{Array, Ix1, Ix2};
use polars::{
    lazy::dsl::col,
    prelude::{DataFrame, IntoLazy},
};

use crate::{io::read_feather_eager, se3::SE3, so3::quat_to_mat3};

/// Pinhole camera intrinsics.
struct Intrinsics {
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

    pub fn k(&self) -> Array<f32, Ix2> {
        let mut k = Array::<f32, Ix2>::eye(2);
        k[[0, 0]] = self.fx_px;
        k[[1, 1]] = self.fy_px;
        k[[0, 2]] = self.cx_px;
        k[[1, 2]] = self.cy_px;
        k
    }
}

struct PinholeCamera {
    /// Pose of camera in the egovehicle frame (inverse of extrinsics matrix).
    ego_se3_cam: SE3,
    /// `Intrinsics` object containing intrinsic parameters and image dimensions.
    intrinsics: Intrinsics,
    /// Associated camera name.
    camera_name: String,
}

impl PinholeCamera {
    /// Return the width of the image in pixels.
    pub fn width_px(&self) -> usize {
        return self.intrinsics.width_px;
    }

    /// Return the height of the image in pixels."""
    pub fn height_px(&self) -> usize {
        return self.intrinsics.height_px;
    }

    /// Return the camera extrinsics.
    pub fn extrinsics(&self) -> Array<f32, Ix2> {
        self.ego_se3_cam.inverse().transform_matrix()
    }

    /// Create a pinhole camera model from a feather file.
    pub fn from_feather(&self, log_dir: &Path, camera_name: &str) -> PinholeCamera {
        let intrinsics_path = log_dir.join("calibration/intrinsics.feather");
        let intrinsics = read_feather_eager(&intrinsics_path, false)
            .lazy()
            .filter(col("sensor_name").eq(camera_name))
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

        let poses_path = log_dir.join("city_SE3_egovehicle.feather");
        let extrinsics = read_feather_eager(&poses_path, false)
            .lazy()
            .filter(col("sensor_name").eq(camera_name))
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
}

fn extract_f32_from_frame(series: &DataFrame, column: &str) -> f32 {
    series[column].get(0).unwrap().try_extract::<f32>().unwrap()
}

fn extract_usize_from_frame(series: &DataFrame, column: &str) -> usize {
    series[column]
        .get(0)
        .unwrap()
        .try_extract::<usize>()
        .unwrap()
}
