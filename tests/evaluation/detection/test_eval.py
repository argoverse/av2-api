# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Detection evaluation unit tests.

Only the last two unit tests here use map ROI information.
The rest apply no filtering to objects that have their corners located outside of the ROI).
"""

import math
from pathlib import Path
from typing import Final, List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from av2.evaluation.detection.constants import AffinityType, DistanceType, TruePositiveErrorNames
from av2.evaluation.detection.eval import evaluate
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    assign,
    compute_affinity_matrix,
    compute_evaluated_dts_mask,
    compute_evaluated_gts_mask,
    distance,
    interpolate_precision,
)
from av2.geometry.geometry import wrap_angles
from av2.geometry.iou import iou_3d_axis_aligned
from av2.utils.constants import PI
from av2.utils.typing import NDArrayBool, NDArrayFloat

TEST_DATA_DIR: Final[Path] = Path(__file__).parent.resolve() / "data"

TRANSLATION_COLS: Final[List[str]] = ["tx_m", "ty_m", "tz_m"]
DIMS_COLS: Final[List[str]] = ["length_m", "width_m", "height_m"]
QUAT_COLS: Final[List[str]] = ["qw", "qx", "qy", "qz"]
ANNO_COLS: Final[List[str]] = ["timestamp_ns", "category"] + DIMS_COLS + QUAT_COLS + TRANSLATION_COLS


def _get_summary_identity() -> pd.DataFrame:
    """Define an evaluator that compares a set of results to itself."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "detections_identity.feather")
    gts: pd.DataFrame = dts.copy()
    gts.loc[:, "num_interior_pts"] = [1, 1, 1, 1, 1, 1]
    _, _, summary = evaluate(dts, gts, detection_cfg)
    return summary


def _get_summary_assignment() -> pd.DataFrame:
    """Define an evaluator that compares a set of results to one with an extra detection to check assignment."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "detections_assignment.feather")
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "labels.feather")
    _, _, summary = evaluate(dts, gts, detection_cfg)
    return summary


def _get_summary() -> pd.DataFrame:
    """Get a dummy summary."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "detections.feather")
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "labels.feather")
    _, _, summary = evaluate(dts, gts, detection_cfg)
    return summary


def test_affinity_center() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated distance matches expected affinity under the specified `AffFnType`.
    """
    columns = DIMS_COLS + QUAT_COLS + TRANSLATION_COLS
    dts = pd.DataFrame([[5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], columns=columns)
    gts = pd.DataFrame([[5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0]], columns=columns)

    expected_result = -5.0
    assert compute_affinity_matrix(dts, gts, AffinityType.CENTER) == expected_result


def test_translation_distance() -> None:
    """Initialize a detection and a ground truth label with only translation parameters.

    Verify that calculated distance matches expected distance under the specified `DistFnType`.
    """
    columns = TRANSLATION_COLS
    dts = pd.DataFrame([[0.0, 0.0, 0.0]], columns=columns)
    gts = pd.DataFrame([[5.0, 5.0, 5.0]], columns=columns)

    expected_result: float = np.sqrt(25 + 25 + 25)
    assert np.allclose(distance(dts, gts, DistanceType.TRANSLATION), expected_result)


def test_scale_distance() -> None:
    """Initialize a detection and a ground truth label with only shape parameters.

    Verify that calculated scale error matches the expected value.

    NOTE: We only provide shape parameters due to alignment assumption.
    """
    columns = DIMS_COLS
    dts = pd.DataFrame([[5.0, 5.0, 5.0]], columns=columns)
    gts = pd.DataFrame([[10.0, 10.0, 10.0]], columns=columns)

    expected_result: float = 1 - 0.125
    assert np.allclose(distance(dts, gts, DistanceType.SCALE), expected_result)


def test_orientation_quarter_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation parameters.

    Verify that calculated orientation error matches the expected smallest angle ((2 * PI) / 4)
    between the detection and ground truth label.
    """
    # Check all of the 90 degree angles
    expected_result: float = (2 * PI) / 4
    quarter_angles: List[NDArrayFloat] = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * PI, expected_result)]

    for i in range(len(quarter_angles) - 1):
        quat_xyzw_dts = Rotation.from_rotvec(quarter_angles[i : i + 1]).as_quat()
        quat_xyzw_gts = Rotation.from_rotvec(quarter_angles[i + 1 : i + 2]).as_quat()

        quat_wxyz_dts = quat_xyzw_dts[..., [3, 0, 1, 2]]
        quat_wxyz_gts = quat_xyzw_gts[..., [3, 0, 1, 2]]

        columns = QUAT_COLS
        dts = pd.DataFrame(quat_wxyz_dts, columns=columns)
        gts = pd.DataFrame(quat_wxyz_gts, columns=columns)

        assert np.isclose(distance(dts, gts, DistanceType.ORIENTATION), expected_result)
        assert np.isclose(distance(gts, dts, DistanceType.ORIENTATION), expected_result)


def test_orientation_eighth_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation parameters.

    Verify that calculated orientation error matches the expected smallest angle ((2 * PI) / 8)
    between the detection and ground truth label.
    """
    expected_result: float = (2 * PI) / 8
    eigth_angles: List[NDArrayFloat] = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * PI, expected_result)]

    for i in range(len(eigth_angles) - 1):
        quat_xyzw_dts = Rotation.from_rotvec(eigth_angles[i : i + 1]).as_quat()
        quat_xyzw_gts = Rotation.from_rotvec(eigth_angles[i + 1 : i + 2]).as_quat()

        quat_wxyz_dts = quat_xyzw_dts[..., [3, 0, 1, 2]]
        quat_wxyz_gts = quat_xyzw_gts[..., [3, 0, 1, 2]]

        columns = QUAT_COLS
        dts = pd.DataFrame(quat_wxyz_dts, columns=columns)
        gts = pd.DataFrame(quat_wxyz_gts, columns=columns)

        assert np.isclose(distance(dts, gts, DistanceType.ORIENTATION), expected_result)
        assert np.isclose(distance(gts, dts, DistanceType.ORIENTATION), expected_result)


def test_wrap_angle() -> None:
    """Test mapping angles to a fixed interval."""
    theta: NDArrayFloat = np.array([-3 * PI / 2])
    expected_result: NDArrayFloat = np.array([PI / 2])
    assert np.isclose(wrap_angles(theta), expected_result)


def test_accumulate() -> None:
    """Verify that the accumulate function matches known output for a self-comparison."""
    cfg = DetectionCfg(eval_only_roi_instances=False)
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "labels.feather")

    for _, group in gts.groupby(["log_id", "timestamp_ns"]):
        job = (group, group, cfg)
        dts, gts = accumulate(*job)

        # Check that there's a true positive under every threshold.
        assert np.all(dts.loc[:, cfg.affinity_thresholds_m])

        # Check that all error metrics are zero.
        assert (dts.loc[:, tuple(x.value for x in TruePositiveErrorNames)] == 0).all(axis=None)

        # Check that there are 2 regular vehicles.
        assert gts["category"].value_counts()["REGULAR_VEHICLE"] == 2

        # Check that there are no other labels.
        assert gts["category"].value_counts().sum() == 2


def test_assign() -> None:
    """Verify that the assign functions as expected by checking ATE of assigned detections against known distance."""
    cfg = DetectionCfg(eval_only_roi_instances=False)

    columns = DIMS_COLS + QUAT_COLS + TRANSLATION_COLS + ["score"]

    dts = pd.DataFrame(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 1.0],
        ],
        columns=columns,
    )

    gts = pd.DataFrame(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 10.1, 10.0, 10.0, 1.0],
        ],
        columns=columns,
    )

    metrics = assign(dts, gts, cfg)
    # if these assign correctly, we should get an ATE of 0.1 for the first two
    expected_result: float = 0.1
    ATE_COL_IDX = 4

    assert math.isclose(metrics.iloc[0, ATE_COL_IDX], expected_result)  # instance 0
    assert math.isclose(metrics.iloc[1, ATE_COL_IDX], expected_result)  # instance 1
    assert math.isclose(metrics.iloc[2, ATE_COL_IDX], 2.0)  # instance 32


def test_interp() -> None:
    """Test non-decreasing `interpolation` constraint enforced on precision results."""
    prec: NDArrayFloat = np.array([1.0, 0.5, 0.33, 0.5])
    expected_result: NDArrayFloat = np.array([1.0, 0.5, 0.5, 0.5])
    assert np.isclose(interpolate_precision(prec), expected_result).all()


def test_iou_aligned_3d() -> None:
    """Initialize a detection and a ground truth label with only shape parameters.

    Verify that calculated intersection-over-union matches the expected
    value between the detection and ground truth label.

    NOTE: We only provide shape parameters due to alignment assumption.
    """
    columns = DIMS_COLS
    dt_dims: NDArrayFloat = pd.DataFrame([[4.0, 10.0, 3.0]], columns=columns).to_numpy()
    gt_dims: NDArrayFloat = pd.DataFrame([[9.0, 5.0, 2.0]], columns=columns).to_numpy()

    # Intersection is 40 = 4 * 5 * 2 (min of all dimensions).
    # Union is the sum of the two volumes, minus intersection: 270 = (10 * 3 * 4) + (5 * 2 * 9) - 40.
    expected_result: float = 40 / 270.0
    assert iou_3d_axis_aligned(dt_dims, gt_dims) == expected_result


def test_assignment() -> None:
    """Verify that assignment works as expected; should have one duplicate in the provided results."""
    expected_result: float = 0.999
    assert _get_summary_assignment().loc["AVERAGE_METRICS", "AP"] == expected_result


def test_ap() -> None:
    """Test that AP is 1 for the self-compared results."""
    expected_result: float = 1.0
    assert _get_summary_identity().loc["AVERAGE_METRICS", "AP"] == expected_result


def test_translation_error() -> None:
    """Test that ATE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.017  # 0.1 / 6, one of six dets is off by 0.1
    assert _get_summary_identity().loc["AVERAGE_METRICS", "ATE"] == expected_result_identity
    assert _get_summary().loc["AVERAGE_METRICS", "ATE"] == expected_result_det


def test_scale_error() -> None:
    """Test that ASE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.033  # 0.2 / 6, one of six dets is off by 20% in IoU
    assert _get_summary_identity().loc["AVERAGE_METRICS", "ASE"] == expected_result_identity
    assert _get_summary().loc["AVERAGE_METRICS", "ASE"] == expected_result_det


def test_orientation_error() -> None:
    """Test that AOE is 0 for the self-compared results."""
    expected_result_identity = 0.0
    expected_result_det = 0.524  # pi / 6, since one of six dets is off by pi

    assert _get_summary_identity().loc["AVERAGE_METRICS", "AOE"] == expected_result_identity
    assert _get_summary().loc["AVERAGE_METRICS", "AOE"] == expected_result_det


def test_compute_evaluated_dts_mask() -> None:
    """Unit test for computing valid detections cuboids."""
    columns = TRANSLATION_COLS + DIMS_COLS + QUAT_COLS
    dts = pd.DataFrame(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # In bounds with at least 1 point.
            [175, 175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # Out of bounds with at least 1 point.
            [-175.0, -175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # Out of bounds with at least 1 point.
            [1.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # In bounds with at least 1 point.
        ],
        columns=columns,
    )
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts_mask = compute_evaluated_dts_mask(dts, detection_cfg)
    dts_mask_: NDArrayBool = np.array([True, False, False, True])
    np.testing.assert_array_equal(dts_mask, dts_mask_)

    dts = pd.DataFrame(np.random.rand(1000, 10), columns=columns)
    dts_mask = compute_evaluated_dts_mask(dts, detection_cfg)
    dts_mask_ = np.zeros(len(dts))
    dts_mask_[: detection_cfg.max_num_dts_per_category] = True
    np.testing.assert_array_equal(dts_mask, dts_mask_)


def test_compute_evaluated_gts_mask() -> None:
    """Unit test for computing valid ground truth cuboids."""
    columns = TRANSLATION_COLS + DIMS_COLS + QUAT_COLS + ["num_interior_pts"]
    gts = pd.DataFrame(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5],  # In bounds with at least 1 point.
            [175, 175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5],  # Out of bounds with at least 1 point.
            [-175.0, -175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5],  # Out of bounds with at least 1 point.
            [1.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0],  # In bounds with at least 1 point.
        ],
        columns=columns,
    )
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    gts_mask = compute_evaluated_gts_mask(gts, detection_cfg)
    gts_mask_: NDArrayBool = np.array([True, False, False, False])
    np.testing.assert_array_equal(gts_mask, gts_mask_)


# def test_filter_objs_to_roi() -> None:
#     """Use the map to filter out an object that lies outside the ROI in a parking lot."""
#     avm = av2Map()

#     # should be outside of ROI
#     outside_obj = {
#         "center": {
#             "x": -14.102872067388489,
#             "y": 19.466695178746022,
#             "z": 0.11740010190455852,
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 0.0,
#             "z": -0.038991328555453404,
#             "w": 0.9992395490058831,
#         },
#         "length": 4.56126567460171,
#         "width": 1.9370055686754908,
#         "height": 1.5820081349372281,
#         "track_label_uuid": "03a321bf955a4d7781682913884abf06",
#         "timestamp": 315970611820366000,
#         "label_class": "VEHICLE",
#     }

#     # should be inside the ROI
#     inside_obj = {
#         "center": {
#             "x": -20.727430239506702,
#             "y": 3.4488006757501353,
#             "z": 0.4036619561689685,
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 0.0,
#             "z": 0.0013102003738908123,
#             "w": 0.9999991416871218,
#         },
#         "length": 4.507580779458834,
#         "width": 1.9243189627993598,
#         "height": 1.629934978730058,
#         "track_label_uuid": "bb0f40e4f68043e285d64a839f2f092c",
#         "timestamp": 315970611820366000,
#         "label_class": "VEHICLE",
#     }

#     log_city_name = "PIT"
#     lidar_ts = 315970611820366000
#     dataset_dir = TEST_DATA_LOC / "roi_based_test"
#     log_id = "21e37598-52d4-345c-8ef9-03ae19615d3d"
#     city_SE3_egovehicle = get_city_SE3_egovehicle_at_sensor_t(lidar_ts, dataset_dir, log_id)

#     dts = np.array([json_label_dict_to_obj_record(item) for item in [outside_obj, inside_obj]])
#     dts_filtered = filter_objs_to_roi(dts, avm, city_SE3_egovehicle, log_city_name)

#     assert dts_filtered.size == 1
#     assert dts_filtered.dtype == "O"  # array of objects
#     assert isinstance(dts_filtered, np.ndarray)
#     assert dts_filtered[0].track_id == "bb0f40e4f68043e285d64a839f2f092c"
