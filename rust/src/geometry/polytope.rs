//! # polygons
//!
//! Geometric algorithms for polygon geometries.

use ndarray::{concatenate, par_azip, s, Array, ArrayView, Axis, Ix1, Ix2, Ix3, Slice};
use once_cell::sync::Lazy;

use super::so3::_quat_to_mat3;

// Safety: 24 elements (8 * 3 = 24) are defined.
static VERTS: Lazy<Array<f32, Ix2>> = Lazy::new(|| unsafe {
    Array::<f32, Ix2>::from_shape_vec_unchecked(
        (8, 3),
        vec![
            1., 1., 1., 1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., -1.,
            -1., -1., -1., 1., -1.,
        ],
    )
});

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

/// Convert (N,10) cuboids to polygons.
pub fn cuboids_to_polygons(cuboids: &ArrayView<f32, Ix2>) -> Array<f32, Ix3> {
    let num_cuboids = cuboids.shape()[0];
    let mut polygons = Array::<f32, Ix3>::zeros([num_cuboids, 8, 3]);
    par_azip!((mut p in polygons.outer_iter_mut(), c in cuboids.outer_iter()) {
        p.assign(&_cuboid_to_polygon(&c))
    });
    polygons
}

/// Convert a single cuboid to a polygon.
fn _cuboid_to_polygon(cuboid: &ArrayView<f32, Ix1>) -> Array<f32, Ix2> {
    let center_xyz = cuboid.slice(s![0..3]);
    let dims_lwh = cuboid.slice(s![3..6]);
    let quat_wxyz = cuboid.slice(s![6..10]);
    let mat = _quat_to_mat3(&quat_wxyz);
    let verts = &VERTS.clone() * &dims_lwh / 2.;
    let verts = verts.dot(&mat.t()) + center_xyz;
    verts.as_standard_layout().to_owned()
}
