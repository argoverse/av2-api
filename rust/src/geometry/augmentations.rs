//! # augmentations
//!
//! Geometric augmentations.

use itertools::Itertools;
use ndarray::{concatenate, Axis};
use polars::{
    lazy::dsl::{col, GetOutput},
    prelude::{DataFrame, DataType, IntoLazy},
    series::Series,
};
use rand_distr::{Bernoulli, Distribution};

use crate::share::{data_frame_to_ndarray_f32, ndarray_to_expr_vec};

use super::so3::{
    reflect_orientation_x, reflect_orientation_y, reflect_translation_x, reflect_translation_y,
};

/// Sample a scene reflection.
/// This reflects both a point cloud and cuboids across the x-axis.
pub fn sample_scene_reflection_x(
    lidar: DataFrame,
    cuboids: DataFrame,
    p: f64,
) -> (DataFrame, DataFrame) {
    let distribution = Bernoulli::new(p).unwrap();
    let is_augmented = distribution.sample(&mut rand::thread_rng());
    if is_augmented {
        let augmented_lidar = lidar
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

        let translation_column_names = vec!["tx_m", "ty_m", "tz_m"];
        let txyz_m = data_frame_to_ndarray_f32(cuboids.clone(), translation_column_names.clone());
        let augmentation_translation = reflect_translation_x(&txyz_m.view());

        let orientation_column_names = vec!["qw", "qx", "qy", "qz"];
        let quat_wxyz =
            data_frame_to_ndarray_f32(cuboids.clone(), orientation_column_names.clone());
        let augmented_orientation = reflect_orientation_x(&quat_wxyz.view());
        let augmented_poses =
            concatenate![Axis(1), augmentation_translation, augmented_orientation];

        let column_names = translation_column_names
            .into_iter()
            .chain(orientation_column_names)
            .collect_vec();
        let series_vec = ndarray_to_expr_vec(augmented_poses, column_names);
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
    let distribution: Bernoulli = Bernoulli::new(p).unwrap();
    let is_augmented = distribution.sample(&mut rand::thread_rng());
    if is_augmented {
        let augmented_lidar = lidar
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

        let translation_column_names = vec!["tx_m", "ty_m", "tz_m"];
        let txyz_m = data_frame_to_ndarray_f32(cuboids.clone(), translation_column_names.clone());
        let augmentation_translation = reflect_translation_y(&txyz_m.view());

        let orientation_column_names = vec!["qw", "qx", "qy", "qz"];
        let quat_wxyz =
            data_frame_to_ndarray_f32(cuboids.clone(), orientation_column_names.clone());
        let augmented_orientation = reflect_orientation_y(&quat_wxyz.view());
        let augmented_poses =
            concatenate![Axis(1), augmentation_translation, augmented_orientation];

        let column_names = translation_column_names
            .into_iter()
            .chain(orientation_column_names)
            .collect_vec();
        let series_vec = ndarray_to_expr_vec(augmented_poses, column_names);
        let augmented_cuboids = cuboids.lazy().with_columns(series_vec).collect().unwrap();
        (augmented_lidar, augmented_cuboids)
    } else {
        (lidar, cuboids)
    }
}
