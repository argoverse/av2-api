//! # augmentations
//!
//! Geometric augmentations.

use std::f32::consts::PI;

use crate::{
    geometry::so3::_mat3_to_quat,
    io::ndarray_from_frame,
    share::{data_frame_to_ndarray_f32, ndarray_to_expr_vec},
};
use itertools::Itertools;
use ndarray::{azip, concatenate, par_azip, s, Array, Axis, Ix1, Ix2};
use polars::{
    lazy::dsl::{col, cols, GetOutput},
    prelude::{DataFrame, DataType, IntoLazy},
    series::Series,
};
use rand_distr::{Bernoulli, Distribution, Uniform};

use super::{
    polytope::{compute_interior_points_mask, cuboids_to_polygons},
    so3::{
        _quat_to_mat3, reflect_orientation_x, reflect_orientation_y, reflect_translation_x,
        reflect_translation_y,
    },
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

/// Sample a scene global scale.
/// This scales the lidar coordinates (x,y,z) and the cuboid centers (tx_m,ty_m,tz_m).
pub fn sample_scene_global_scale(
    lidar: DataFrame,
    cuboids: DataFrame,
    low_inclusive: f64,
    upper_inclusive: f64,
) -> (DataFrame, DataFrame) {
    let distribution = Uniform::new_inclusive(low_inclusive, upper_inclusive);
    let scale_factor = distribution.sample(&mut rand::thread_rng()) as f32;
    let augmented_lidar = lidar
        .lazy()
        .with_column(col("x").map(
            move |x| {
                Ok(Some(
                    x.f32()
                        .unwrap()
                        .into_no_null_iter()
                        .map(|x| scale_factor * x)
                        .collect::<Series>(),
                ))
            },
            GetOutput::from_type(DataType::Float32),
        ))
        .with_column(col("y").map(
            move |y| {
                Ok(Some(
                    y.f32()
                        .unwrap()
                        .into_no_null_iter()
                        .map(|y| scale_factor * y)
                        .collect::<Series>(),
                ))
            },
            GetOutput::from_type(DataType::Float32),
        ))
        .with_column(col("z").map(
            move |z| {
                Ok(Some(
                    z.f32()
                        .unwrap()
                        .into_no_null_iter()
                        .map(|z| scale_factor * z)
                        .collect::<Series>(),
                ))
            },
            GetOutput::from_type(DataType::Float32),
        ))
        .collect()
        .unwrap();

    let augmented_cuboids = cuboids
        .lazy()
        .with_column(col("tx_m").map(
            move |x| {
                Ok(Some(
                    x.f32()
                        .unwrap()
                        .into_no_null_iter()
                        .map(|x| scale_factor * x)
                        .collect::<Series>(),
                ))
            },
            GetOutput::from_type(DataType::Float32),
        ))
        .with_column(col("ty_m").map(
            move |y| {
                Ok(Some(
                    y.f32()
                        .unwrap()
                        .into_no_null_iter()
                        .map(|y| scale_factor * y)
                        .collect::<Series>(),
                ))
            },
            GetOutput::from_type(DataType::Float32),
        ))
        .with_column(col("tz_m").map(
            move |z| {
                Ok(Some(
                    z.f32()
                        .unwrap()
                        .into_no_null_iter()
                        .map(|z| scale_factor * z)
                        .collect::<Series>(),
                ))
            },
            GetOutput::from_type(DataType::Float32),
        ))
        .collect()
        .unwrap();
    (augmented_lidar, augmented_cuboids)
}

/// Sample a scene global rotation.
/// This rotates the lidar coordinates (x,y,z) and the cuboid centers (tx_m,ty_m,tz_m).
pub fn sample_scene_global_rotation(
    lidar: DataFrame,
    cuboids: DataFrame,
    low_inclusive: f64,
    upper_inclusive: f64,
) -> (DataFrame, DataFrame) {
    let distribution = Uniform::new_inclusive(low_inclusive, upper_inclusive);
    let theta = distribution.sample(&mut rand::thread_rng()) as f32;
    let rotation = Array::<f32, Ix1>::from_vec(vec![
        f32::cos(2.0 * PI * theta),
        f32::sin(2.0 * PI * theta),
        0.0,
        -f32::sin(2.0 * PI * theta),
        f32::cos(2.0 * PI * theta),
        0.0,
        0.0,
        0.0,
        1.0,
    ])
    .into_shape((3, 3))
    .unwrap();

    let column_names = ["x", "y", "z"];
    let lidar_ndarray = ndarray_from_frame(&lidar, cols(column_names));
    let augmented_lidar_ndarray = lidar_ndarray.dot(&rotation);
    let series_vec = ndarray_to_expr_vec(augmented_lidar_ndarray, column_names.to_vec());
    let augmented_lidar = lidar.lazy().with_columns(series_vec).collect().unwrap();

    let cuboid_column_names = ["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"];
    let cuboids_ndarray = ndarray_from_frame(&cuboids, cols(cuboid_column_names));

    let num_cuboids = cuboids_ndarray.shape()[0];
    let mut augmented_cuboids = Array::<f32, Ix2>::zeros((num_cuboids, 7));
    par_azip!((mut ac in augmented_cuboids.outer_iter_mut(), c in cuboids_ndarray.outer_iter()) {
        let augmented_translation = c.slice(s![..3]).dot(&rotation);
        let augmented_mat3 = _quat_to_mat3(&c.slice(s![3..7])).dot(&rotation.t());
        let augmented_rotation = _mat3_to_quat(&augmented_mat3.view());

        ac.slice_mut(s![..3]).assign(&augmented_translation);
        ac.slice_mut(s![3..7]).assign(&augmented_rotation);
    });

    let series_vec = ndarray_to_expr_vec(augmented_cuboids, cuboid_column_names.to_vec());
    let augmented_cuboids = cuboids.lazy().with_columns(series_vec).collect().unwrap();
    (augmented_lidar, augmented_cuboids)
}

/// Sample a scene with random object scaling.
pub fn sample_random_object_scale(
    lidar: DataFrame,
    cuboids: DataFrame,
    low_inclusive: f64,
    high_inclusive: f64,
) -> (DataFrame, DataFrame) {
    let cuboid_column_names = [
        "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz",
    ];
    let mut lidar_ndarray = ndarray_from_frame(&lidar, cols(["x", "y", "z"]));
    let mut cuboids_ndarray = ndarray_from_frame(&cuboids, cols(cuboid_column_names));
    let cuboid_vertices = cuboids_to_polygons(&cuboids_ndarray.view());
    let interior_points_mask =
        compute_interior_points_mask(&lidar_ndarray.view(), &cuboid_vertices.view());

    let distribution = Uniform::new_inclusive(low_inclusive, high_inclusive);

    azip!((mut c in cuboids_ndarray.outer_iter_mut(), m in interior_points_mask.outer_iter()) {
        let scale_factor = distribution.sample(&mut rand::thread_rng()) as f32;
        let indices = m
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match *x {
                true => Some(i),
                _ => None,
            })
            .collect_vec();
        let interior_points = lidar_ndarray.select(Axis(0), &indices);

        // TODO: Handle with iterators.
        for (i, j) in indices.into_iter().enumerate() {
            let scaled_lidar_ndarray_object = (&interior_points.slice(s![i, ..])
                - &c.slice(s![..3]))
                * scale_factor;
            let scaled_lidar_ndarray_egovehicle =
                &scaled_lidar_ndarray_object + &c.slice(s![..3]);

            lidar_ndarray
                .slice_mut(s![j, ..])
                .assign(&scaled_lidar_ndarray_egovehicle);
        }
        c.slice_mut(s![3..6]).mapv_inplace(|x| x * scale_factor);
    });

    let lidar_column_names = vec!["x", "y", "z"];
    let series_vec = ndarray_to_expr_vec(lidar_ndarray, lidar_column_names);
    let augmented_lidar = lidar.lazy().with_columns(series_vec).collect().unwrap();

    let series_vec = ndarray_to_expr_vec(cuboids_ndarray, cuboid_column_names.to_vec());
    let augmented_cuboids = cuboids.lazy().with_columns(series_vec).collect().unwrap();
    (augmented_lidar, augmented_cuboids)
}
