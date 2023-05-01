# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for polyline interpolation utilities."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import av2.geometry.interpolate as interp_utils
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayFloat


def test_compute_lane_width_straight() -> None:
    """Compute the lane width of the following straight lane segment.

    (waypoints indicated with "o" symbol):

            o   o
            |   |
            o   o
            |   |
            o   o

    We can swap boundaries for this lane, and the width should be identical.
    """
    left_even_pts: NDArrayFloat = np.array([[1, 1], [1, 0], [1, -1]])
    right_even_pts: NDArrayFloat = np.array([[-1, 1], [-1, 0], [-1, -1]])
    lane_width = interp_utils.compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = 2.0
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = interp_utils.compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = 2.0
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_lane_width_telescoping() -> None:
    r"""Compute the lane width of the following straight lane segment.

    (waypoints indicated with "o" symbol):

    right       left
     o           o
     \\         //
      o        o
       \\     //
        o     o
         \\ //
           o

    We can swap boundaries for this lane, and the width should be identical.
    """
    # fmt: off
    left_even_pts: NDArrayFloat = np.array(
        [
            [3,  2],
            [2,  1],
            [1,  0],
            [0, -1]
        ]
    )
    right_even_pts: NDArrayFloat = np.array(
        [
            [-3,  2],
            [-2,  1],
            [-1,  0],
            [ 0, -1]
        ]
    )
    # fmt: on
    lane_width = interp_utils.compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = (6.0 + 4.0 + 2.0 + 0.0) / 4
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = interp_utils.compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = (6.0 + 4.0 + 2.0 + 0.0) / 4
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_lane_width_curved_width1() -> None:
    r"""Compute the lane width of the following curved lane segment.

    Should have width 1 at each pair of boundary waypoints.

          -------boundary
         /  ----boundary
        /  /
        |  |
        |  \\
         \\ -----
          \\-----
    """
    left_even_pts: NDArrayFloat = np.array(
        [[0, 2], [-2, 2], [-3, 1], [-3, 0], [-2, -1], [0, -1]]
    )
    right_even_pts: NDArrayFloat = np.array(
        [[0, 3], [-2, 3], [-4, 1], [-4, 0], [-2, -2], [0, -2]]
    )
    lane_width = interp_utils.compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = 1.0
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = interp_utils.compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = 1.0
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_lane_width_curved_not_width1() -> None:
    r"""Compute the lane width of the following curved lane segment.

          -------
         /  ----
        /  /
        |  |
        |  \\
         \\ -----
          \\-----

    We get waypoint distances of [1,1,1,1,0.707..., 1,1]
    """
    left_even_pts: NDArrayFloat = np.array(
        [[0, 2], [-2, 2], [-3, 1], [-3, 0], [-2.5, -0.5], [-2, -1], [0, -1]]
    )

    right_even_pts: NDArrayFloat = np.array(
        [[0, 3], [-2, 3], [-4, 1], [-4, 0], [-3, -1], [-2, -2], [0, -2]]
    )

    lane_width = interp_utils.compute_lane_width(left_even_pts, right_even_pts)
    gt_lane_width = 0.9581581115980783
    assert np.isclose(lane_width, gt_lane_width)

    lane_width = interp_utils.compute_lane_width(right_even_pts, left_even_pts)
    gt_lane_width = 0.9581581115980783
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_mid_pivot_arc_3pt_cul_de_sac() -> None:
    """Make sure we handle the cul-de-sac case correctly, for 2d polylines.

    When mapping a cul-de-sac, we get a line of points on one boundary,
    and a single point on the other side. This function produces the middle
    arc we get by pivoting around the single point.

    Waypoints are depicted below for the cul-de-sac center, and other boundary.

            o
             \
              \
               \
            O   o
               /
              /
             /
            o
    """
    # Numpy array of shape (3,)
    single_pt: NDArrayFloat = np.array([0, 0])

    # Numpy array of shape (N,3)
    arc_pts: NDArrayFloat = np.array([[0, 1], [1, 0], [0, -1]])

    # centerline_pts: Numpy array of shape (N,3)
    centerline_pts, lane_width = interp_utils.compute_mid_pivot_arc(single_pt, arc_pts)

    gt_centerline_pts: NDArrayFloat = np.array([[0, 0.5], [0.5, 0], [0, -0.5]])
    gt_lane_width = 1.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_mid_pivot_arc_3pt_cul_de_sac_3dpolylines() -> None:
    """Make sure we handle the cul-de-sac case correctly, for 3d polylines.

    When mapping a cul-de-sac, we get a line of points on one boundary,
    and a single point on the other side. This function produces the middle
    arc we get by pivoting around the single point.

    Waypoints are depicted below for the cul-de-sac center, and other boundary.
    The scenario is a banked turn, where the inside is low, and the outside polyline is higher.

            o @ (0,1)
              \
            .   \
                  \
            O  .   o  @ (1,0)
                  /
            .   /
              /
            o @ (0,-1)
    """
    # Numpy array of shape (3,)
    single_pt: NDArrayFloat = np.array([0, 0, 0])

    # Numpy array of shape (N,3)
    # fmt: off
    arc_pts: NDArrayFloat = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [0, -1, 2]
        ]
    )
    # fmt: on

    # centerline_pts: Numpy array of shape (N,3)
    centerline_pts, lane_width = interp_utils.compute_mid_pivot_arc(single_pt, arc_pts)

    # z values should be halfway between outer arc and inner arc
    gt_centerline_pts: NDArrayFloat = np.array([[0, 0.5, 1], [0.5, 0, 1], [0, -0.5, 1]])
    assert np.allclose(centerline_pts, gt_centerline_pts)

    # compute hypotenuse using height 2 and width 1 -> 2^2 + 1^2 = 5^2
    gt_lane_width = np.sqrt(5)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_mid_pivot_arc_5pt_cul_de_sac() -> None:
    """Make sure we handle the cul-de-sac case correctly, for 2d polylines.

    When mapping a cul-de-sac, we get a line of points on one boundary,
    and a single point on the other side. This function produces the middle
    arc we get by pivoting around the single point.

    Waypoints are depicted below for the cul-de-sac center, and other boundary.

            o
             \
              o
               \
            O   o
               /
              o
             /
            o
    """
    # Numpy array of shape (3,)
    single_pt: NDArrayFloat = np.array([0, 0])

    # Numpy array of shape (N, 3)
    arc_pts: NDArrayFloat = np.array([[0, 2], [1, 1], [2, 0], [1, -1], [0, -2]])

    # centerline_pts: Numpy array of shape (N,3)
    centerline_pts, lane_width = interp_utils.compute_mid_pivot_arc(single_pt, arc_pts)

    gt_centerline_pts: NDArrayFloat = np.array(
        [[0, 1], [0.5, 0.5], [1, 0], [0.5, -0.5], [0, -1]]
    )
    gt_lane_width = (2 + 2 + 2 + np.sqrt(2) + np.sqrt(2)) / 5
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_cul_de_sac_right_onept() -> None:
    """Compute midpoint line for a cul-de-sac with one right waypoint.

    Make sure that if we provide left and right boundary polylines,
    we can get the correct centerline by averaging left and right waypoints.
    """
    left_ln_bnds: NDArrayFloat = np.array([[0, 2], [1, 1], [2, 0], [1, -1], [0, -2]])
    right_ln_bnds: NDArrayFloat = np.array([[0, 0]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=5
    )

    gt_centerline_pts: NDArrayFloat = np.array(
        [[0, 1], [0.5, 0.5], [1, 0], [0.5, -0.5], [0, -1]]
    )
    gt_lane_width = (2 + 2 + 2 + np.sqrt(2) + np.sqrt(2)) / 5

    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_cul_de_sac_left_onept() -> None:
    """Compute midpoint line for a cul-de-sac with one left waypoint.

    Make sure that if we provide left and right boundary polylines in 2d,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds: NDArrayFloat = np.array([[0, 2], [1, 1], [2, 0], [1, -1], [0, -2]])
    left_ln_bnds: NDArrayFloat = np.array([[0, 0]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=5
    )

    gt_centerline_pts: NDArrayFloat = np.array(
        [[0, 1], [0.5, 0.5], [1, 0], [0.5, -0.5], [0, -1]]
    )
    gt_lane_width = (2 + 2 + 2 + np.sqrt(2) + np.sqrt(2)) / 5

    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_5_waypts() -> None:
    """Compute midpoint line for a straightline with five waypoints.

    Make sure that if we provide left and right boundary polylines in 2d,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds: NDArrayFloat = np.array(
        [[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]]
    )
    left_ln_bnds: NDArrayFloat = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=5
    )

    gt_centerline_pts: NDArrayFloat = np.array(
        [[0.5, 4], [0.5, 2], [0.5, 0], [0.5, -2], [0.5, -4]]
    )
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_4_waypts() -> None:
    """Test computing midpoint line with 4 waypoints.

    Make sure that if we provide left and right boundary polylines in 2d,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds: NDArrayFloat = np.array(
        [[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]]
    )
    left_ln_bnds: NDArrayFloat = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=4
    )

    gt_centerline_pts: NDArrayFloat = np.array(
        [[0.5, 4], [0.5, 4 / 3], [0.5, -4 / 3], [0.5, -4]]
    )
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_3_waypts() -> None:
    """Test computing midpoint line with 3 waypoints.

    Make sure that if we provide left and right boundary polylines in 2d,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds: NDArrayFloat = np.array(
        [[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]]
    )
    left_ln_bnds: NDArrayFloat = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=3
    )

    gt_centerline_pts: NDArrayFloat = np.array([[0.5, 4], [0.5, 0], [0.5, -4]])
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_straightline_maintain_2_waypts() -> None:
    """Test computing midpoint line with 2 waympoints.

    Make sure that if we provide left and right boundary polylines in 2d,
    we can get the correct centerline by averaging left and right waypoints.
    """
    right_ln_bnds: NDArrayFloat = np.array(
        [[-1, 4], [-1, 2], [-1, 0], [-1, -2], [-1, -4]]
    )
    left_ln_bnds: NDArrayFloat = np.array([[2, 4], [2, 2], [2, 0], [2, -2], [2, -4]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=2
    )

    gt_centerline_pts: NDArrayFloat = np.array([[0.5, 4], [0.5, -4]])
    gt_lane_width = 3.0
    assert np.allclose(centerline_pts, gt_centerline_pts)
    assert np.isclose(lane_width, gt_lane_width)


def test_compute_midpoint_line_curved_maintain_4_waypts() -> None:
    """Test computing midpoint line.

    Make sure that if we provide left and right boundary polylines in 2d,
    we can get the correct centerline by averaging left and right waypoints.

    Note that because of the curve and the arc interpolation, the land width and centerline in the middle points
    will be shifted.
    """
    right_ln_bnds: NDArrayFloat = np.array([[-1, 3], [1, 3], [4, 0], [4, -2]])
    left_ln_bnds: NDArrayFloat = np.array([[-1, 1], [1, 1], [2, 0], [2, -2]])

    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=4
    )

    # from argoverse.utils.mpl_plotting_utils import draw_polygon_mpl

    # fig = plt.figure(figsize=(22.5, 8))
    # ax = fig.add_subplot(111)

    # draw_polygon_mpl(ax, right_ln_bnds, "g")
    # draw_polygon_mpl(ax, left_ln_bnds, "b")
    # draw_polygon_mpl(ax, centerline_pts, "r")

    gt_centerline_pts: NDArrayFloat = np.array([[-1, 2], [1, 2], [3, 0], [3, -2]])

    assert np.allclose(centerline_pts[0], gt_centerline_pts[0])
    assert np.allclose(centerline_pts[-1], gt_centerline_pts[-1])


def test_compute_midpoint_line_straightline_maintain_3_waypts_3dpolylines() -> None:
    """Test computing correct centerline.

    Make sure that if we provide left and right boundary polylines in 3d,
    we can get the correct centerline by averaging left and right waypoints.
    """
    # fmt: off
    # right side is lower, at z = 0
    right_ln_bnds: NDArrayFloat = np.array(
        [
            [-1,  4, 0],
            [-1,  2, 0],
            [-1,  0, 0],
            [-1, -2, 0],
            [-1, -4, 0]
        ]
    )
    # left side is higher, at z = 2
    # fewer waypoint on left side, but shouldn't affect anything for a straight line
    left_ln_bnds: NDArrayFloat = np.array(
        [
            [2,  4, 2],
            [2, -2, 2],
            [2, -4, 2]
        ]
    )

    # fmt: on
    centerline_pts, lane_width = interp_utils.compute_midpoint_line(
        left_ln_bnds, right_ln_bnds, num_interp_pts=3
    )
    # fmt: off
    gt_centerline_pts: NDArrayFloat = np.array(
        [
            [0.5,  4, 1],
            [0.5,  0, 1],
            [0.5, -4, 1]
        ]
    )
    # fmt: on
    assert np.allclose(centerline_pts, gt_centerline_pts)

    # hypotenuse from width=3 on xy plane and z height diff of 3 -> 3**2 + 2**2 = 13
    gt_lane_width = np.sqrt(13)
    assert np.isclose(lane_width, gt_lane_width)


def test_interp_arc_straight_line() -> None:
    """Test arc interpolation on a straight line."""
    pts: NDArrayFloat = np.array([[-10, 0], [10, 0]])
    interp_pts = interp_utils.interp_arc(t=3, points=pts)
    # fmt: off
    gt_interp_pts: NDArrayFloat = np.array(
        [
            [-10, 0],
            [  0, 0],
            [ 10, 0]
        ]
    )
    # fmt: on
    assert np.allclose(interp_pts, gt_interp_pts)

    interp_pts = interp_utils.interp_arc(t=4, points=pts)
    # fmt: off
    gt_interp_pts_: NDArrayFloat = np.array(
        [
            [-10,     0],
            [-10 / 3, 0],
            [ 10 / 3, 0],
            [ 10,     0]
        ]
    )
    # fmt: on
    assert np.allclose(interp_pts, gt_interp_pts_)


def test_interp_arc_straight_line_3d() -> None:
    """Ensure that linear interpolation works in 3d."""
    # fmt: off
    pts: NDArrayFloat = np.array(
        [
            [-10, 0, -1],
            [ 10, 0,  1],
        ]
    )
    # fmt: on
    interp_pts = interp_utils.interp_arc(t=3, points=pts)

    # expect to get 3 waypoints along the straight line
    # fmt: off
    expected_interp_pts: NDArrayFloat = np.array(
        [
            [-10.,   0.,  -1.],
            [  0.,   0.,   0.],
            [ 10.,   0.,   1.]
        ]
    )
    # fmt: on
    assert np.allclose(interp_pts, expected_interp_pts)


def test_interp_arc_straight_line_3d_5pts() -> None:
    """Ensure that linear interpolation works in 3d, with 5 desired waypoints."""
    # fmt: off
    pts: NDArrayFloat = np.array(
        [
            [-10, 0, -1],
            [ 10, 0,  1],
        ]
    )
    # fmt: on
    interp_pts = interp_utils.interp_arc(t=5, points=pts)

    # expect to get 5 waypoints along the straight line
    # fmt: off
    expected_interp_pts: NDArrayFloat = np.array(
        [
            [-10, 0, -1],
            [ -5, 0, -0.5],
            [  0, 0,  0],
            [  5, 0,  0.5],
            [ 10, 0,  1]
        ]
    )
    # fmt: on
    assert np.allclose(interp_pts, expected_interp_pts)


def test_interp_arc_curved_line_3d_5pts() -> None:
    r"""Ensure that linear interpolation works for curves in 3d, with 5 desired waypoints.

    Shape of interpolated polyline:
        .        +       .
         \\           //
            .        .
              \\   //
                 .
    """
    # fmt: off
    pts: NDArrayFloat = np.array(
        [
            [-10,   0, -1],
            [  0, -10,  0.5],
            [ 10,   0,  1],
        ]
    )
    # fmt: on
    interp_pts = interp_utils.interp_arc(t=5, points=pts)

    # expect to get 5 waypoints along the straight line
    # fmt: off
    expected_interp_pts: NDArrayFloat = np.array(
        [
            [-10,   0, -1],
            [ -5,  -5, -0.25],
            [  0, -10,  0.5],
            [  5,  -5,  0.75],
            [ 10,   0,  1]
        ]
    )
    # fmt: on
    assert np.allclose(interp_pts, expected_interp_pts, atol=0.03)


def test_interpolate_pose() -> None:
    """Interpolate poses between the time interval [0,1].

    Note: visualization may be turned on to understand the setup. See figure here:
        https://user-images.githubusercontent.com/16724970/155865211-cee5f72c-a886-4d47-bc75-65feb3b41fe6.png

    Two coordinate frames @t0 and @t1 are visualized in 2d below:

    @t1
       |
    ___|
       .
       .     |
    .........|___ @t0
       .
       .
    """
    visualize = False

    city_SE3_egot0 = SE3(rotation=np.eye(3), translation=np.array([5, 0, 0]))
    city_SE3_egot1 = SE3(
        rotation=Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        translation=np.array([0, 5, 0]),
    )

    t0 = 0
    t1 = 10
    for query_timestamp in np.arange(11):
        pose = interp_utils.interpolate_pose(
            key_timestamps=(t0, t1),
            key_poses=(city_SE3_egot0, city_SE3_egot1),
            query_timestamp=query_timestamp,
        )
        if visualize:
            _plot_pose(pose)
    if visualize:
        plt.axis("equal")
        plt.show()


def _plot_pose(cTe: SE3) -> None:
    """Visualize a pose by plotting x,y axes of a coordinate frame in red and green.

    Args:
        cTe: represents an egovehicle pose in the city frame.
    """
    x0_e: NDArrayFloat = np.array([[0, 0, 0]], dtype=float)
    x0_c = cTe.transform_point_cloud(x0_e).squeeze()

    x1_e: NDArrayFloat = np.array([[1, 0, 0]], dtype=float)
    x1_c = cTe.transform_point_cloud(x1_e).squeeze()

    x2_e: NDArrayFloat = np.array([[0, 1, 0]], dtype=float)
    x2_c = cTe.transform_point_cloud(x2_e).squeeze()

    plt.plot([x0_c[0], x1_c[0]], [x0_c[1], x1_c[1]], color="r")
    plt.plot([x0_c[0], x2_c[0]], [x0_c[1], x2_c[1]], color="g")


def test_linear_interpolation() -> None:
    """Ensure we can linear interpolation positions in an interval [t0,t1].

    Locations marked by "O" below:
    @t1
       |
     O |
       \
    ---|-O--
       | @ t0
    """
    X0: NDArrayFloat = np.array([1, 0, 0], dtype=float)
    X1: NDArrayFloat = np.array([-1, 2, 10], dtype=float)

    # at start of interval (@5 sec)
    Xt_5 = interp_utils.linear_interpolation(
        key_timestamps=(5, 15), key_translations=(X0, X1), query_timestamp=5
    )
    expected_Xt_5: NDArrayFloat = np.array([1, 0, 0], dtype=float)
    assert np.array_equal(Xt_5, expected_Xt_5)

    # midway through interval (@10 sec)
    Xt_10 = interp_utils.linear_interpolation(
        key_timestamps=(5, 15), key_translations=(X0, X1), query_timestamp=10
    )
    expected_Xt_10: NDArrayFloat = np.array([0, 1, 5], dtype=float)
    assert np.array_equal(Xt_10, expected_Xt_10)

    # at end of interval (@15 sec)
    Xt_15 = interp_utils.linear_interpolation(
        key_timestamps=(5, 15), key_translations=(X0, X1), query_timestamp=15
    )
    expected_Xt_15: NDArrayFloat = np.array([-1, 2, 10], dtype=float)
    assert np.array_equal(Xt_15, expected_Xt_15)
