//! # polygons
//!
//! Geometric algorithms for polygon geometries.

use ndarray::{concatenate, par_azip, s, Array, ArrayView, Axis, Ix2, Ix3, Slice};

/// Compute a boolean mask indicating which points are interior to the cuboid geometry.
pub fn compute_interior_points_mask(
    points: &ArrayView<f32, Ix2>,
    cuboid_vertices: &ArrayView<f32, Ix3>,
) -> Array<bool, Ix2> {
    let num_points = points.shape()[0];
    let num_cuboids = cuboid_vertices.shape()[0];

    let a = cuboid_vertices.slice_axis(Axis(1), Slice::from(6..7));
    let b = cuboid_vertices.slice_axis(Axis(1), Slice::from(3..4));
    let c = cuboid_vertices.slice_axis(Axis(1), Slice::from(1..2));
    let vertices = concatenate![Axis(1), a, b, c];

    let reference_index = cuboid_vertices
        .slice_axis(Axis(1), Slice::from(2..3))
        .to_owned();

    let uvw = reference_index.clone() - vertices.clone();
    let reference_index = reference_index.into_shape((num_cuboids, 3)).unwrap();

    let mut dot_uvw_reference = Array::<f32, Ix2>::zeros((num_cuboids, 3));
    par_azip!((mut a in dot_uvw_reference.outer_iter_mut(), b in uvw.outer_iter(), c in reference_index.outer_iter()) a.assign(&b.dot(&c.t())) );

    let mut dot_uvw_vertices = Array::<f32, Ix2>::zeros((num_cuboids, 3));
    par_azip!((mut a in dot_uvw_vertices.outer_iter_mut(), b in uvw.outer_iter(), c in vertices.outer_iter()) a.assign(&b.dot(&c.t()).diag()) );

    let dot_uvw_points = uvw
        .into_shape((num_cuboids * 3, 3))
        .unwrap()
        .as_standard_layout()
        .dot(&points.t().as_standard_layout())
        .into_shape((num_cuboids, 3, num_points))
        .unwrap();

    let shape = (num_cuboids, num_points);
    let mut is_interior =
        Array::<_, Ix2>::from_shape_vec(shape, vec![false; num_cuboids * num_points]).unwrap();
    par_azip!((mut a in is_interior.outer_iter_mut(), b in dot_uvw_reference.outer_iter(), c in dot_uvw_points.outer_iter(), d in dot_uvw_vertices.outer_iter()) {

        let c0 = c.slice(s![0, ..]).mapv(|x| ((b[0] <= x) & (x <= d[0])) | ((b[0] >= x) & (x >= d[0])));
        let c1 = c.slice(s![1, ..]).mapv(|x| ((b[1] <= x) & (x <= d[1])) | ((b[1] >= x) & (x >= d[1])));
        let c2 = c.slice(s![2, ..]).mapv(|x| ((b[2] <= x) & (x <= d[2])) | ((b[2] >= x) & (x >= d[2])));

        let is_interior_i = &c0 & &c1 & &c2;
        a.assign(&is_interior_i);
    });

    is_interior
}
