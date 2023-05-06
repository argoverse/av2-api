//! # augmentations
//!
//! Geometric augmentations.

use ndarray::{Array, ArrayView, Ix2};
use polars::{
    lazy::dsl::{col, GetOutput},
    prelude::{DataFrame, DataType, IntoLazy},
    series::Series,
};
use rand_distr::{Bernoulli, Distribution};

use crate::share::{data_frame_to_ndarray_f32, ndarray_to_series_vec};

use super::se3::{reflect_pose_x, reflect_pose_y};

/// Reflect all poses across the x-axis with probability `p` (sampled from a Bernoulli distribution).
pub fn _sample_global_reflect_pose_x(xyzlwh_qwxyz: ArrayView<f32, Ix2>, p: f64) -> Array<f32, Ix2> {
    let d = Bernoulli::new(p).unwrap();
    if d.sample(&mut rand::thread_rng()) {
        return reflect_pose_x(&xyzlwh_qwxyz.view());
    }
    xyzlwh_qwxyz.to_owned()
}

/// Reflect all poses across the y-axis with probability `p` (sampled from a Bernoulli distribution).
pub fn _sample_global_reflect_pose_y(xyzlwh_qwxyz: ArrayView<f32, Ix2>, p: f64) -> Array<f32, Ix2> {
    let d = Bernoulli::new(p).unwrap();
    if d.sample(&mut rand::thread_rng()) {
        return reflect_pose_y(&xyzlwh_qwxyz.view());
    }
    xyzlwh_qwxyz.to_owned()
}

/// Sample a scene reflection.
/// This reflects both a point cloud and cuboids across the x-axis.
pub fn sample_scene_reflection_x(
    lidar: DataFrame,
    cuboids: DataFrame,
    p: f64,
) -> (DataFrame, DataFrame) {
    let d = Bernoulli::new(p).unwrap();
    let is_augmented = d.sample(&mut rand::thread_rng());
    if is_augmented {
        let augmented_lidar = lidar
            .clone()
            .lazy()
            .with_column(col("y").map(
                move |x| {
                    Ok(Some(
                        x.f32()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|y| -y)
                            .collect::<Series>(),
                    ))
                },
                GetOutput::from_type(DataType::Float32),
            ))
            .collect()
            .unwrap();

        // println!("{}", augmented_lidar);

        let column_names = vec!["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"];
        let poses = data_frame_to_ndarray_f32(cuboids.clone(), column_names.clone());
        let augmented_poses = reflect_pose_x(&poses.view());
        let series_vec = ndarray_to_series_vec(augmented_poses, column_names);
        let augmented_cuboids = cuboids.lazy().with_columns(series_vec).collect().unwrap();
        (augmented_lidar, augmented_cuboids)
    } else {
        (lidar, cuboids)
    }
}

/// Sample a scene reflection.
/// This reflects both a point cloud and cuboids across the y-axis.
pub fn sample_scene_reflection_y(
    lidar: DataFrame,
    cuboids: DataFrame,
    p: f64,
) -> (DataFrame, DataFrame) {
    let d = Bernoulli::new(p).unwrap();
    let is_augmented = d.sample(&mut rand::thread_rng());
    if is_augmented {
        let augmented_lidar = lidar
            .clone()
            .lazy()
            .with_column(col("x").map(
                move |x| {
                    Ok(Some(
                        x.f32()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|x| -x)
                            .collect::<Series>(),
                    ))
                },
                GetOutput::from_type(DataType::Float32),
            ))
            .collect()
            .unwrap();

        let column_names = vec!["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"];
        let poses = data_frame_to_ndarray_f32(cuboids.clone(), column_names.clone());
        let augmented_poses = reflect_pose_y(&poses.view());
        let series_vec = ndarray_to_series_vec(augmented_poses, column_names);
        let augmented_cuboids = cuboids.lazy().with_columns(series_vec).collect().unwrap();
        (augmented_lidar, augmented_cuboids)
    } else {
        (lidar, cuboids)
    }
}
