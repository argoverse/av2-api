"""Scene Flow evaluation unit tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import av2.evaluation.scene_flow.constants as constants
import av2.evaluation.scene_flow.eval as eval
from av2.evaluation.scene_flow.make_annotation_files import write_annotation
from av2.evaluation.scene_flow.utils import get_eval_point_mask, write_output_file
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

gts: NDArrayFloat = np.array(
    [
        [0.0, 0.0, 0.0],
        [1e-3, 0.0, 0.0],
        [0.0, 1e-3, 0.0],
        [0.0, 0.0, 1e-3],
        [1e3, 0.0, 0.0],
        [0.0, 1e3, 0.0],
        [0.0, 0.0, 1e3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

dts_perfect: NDArrayFloat = gts.copy()
dts_very_close: NDArrayFloat = gts + 0.01
dts_close: NDArrayFloat = gts + 0.05
dts_zero: NDArrayFloat = np.zeros_like(gts).astype(np.float64)

gts_dynamic: NDArrayBool = np.array([False, False, False, False, True, True, True, True, False, False])
gts_classes: NDArrayInt = np.array([0, 0, 0, 0, 17, 17, 1, 11, 0, 23])
gts_valid: NDArrayBool = np.array([False, True, True, True, True, True, True, True, True, True])
gts_close: NDArrayBool = np.array([True, True, False, False, True, True, False, False, True, True])

dts_dynamic: NDArrayBool = np.array([False, True, False, True, False, True, False, True, False, True])


def test_end_point_error() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated end-point error matches the expected value.
    """
    epe_perfect = np.zeros(10)
    epe_very_close = (np.sqrt(0.01**2 * 3, dtype=np.float64) * np.ones(10)).astype(np.float64)
    epe_close = (np.sqrt(0.05**2 * 3, dtype=np.float64) * np.ones(10)).astype(np.float64)
    epe_zero = np.array([0.0e00, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e3, 1.0e3, 1.0e3, 1.0e00, 1.0e00, 1.0e00], dtype=np.float64)

    assert np.allclose(eval.compute_end_point_error(dts_perfect, gts), epe_perfect)
    assert np.allclose(eval.compute_end_point_error(dts_very_close, gts), epe_very_close)
    assert np.allclose(eval.compute_end_point_error(dts_close, gts), epe_close)
    assert np.allclose(eval.compute_end_point_error(dts_zero, gts), epe_zero)


def test_accuracy_strict() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated strict accuracy matches the expected value.
    """
    accS_perfect = np.ones(10)
    accS_close = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    accS_very_close = np.ones(10)
    accS_zero = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert np.allclose(eval.compute_accuracy_strict(dts_perfect, gts), accS_perfect)
    assert np.allclose(eval.compute_accuracy_strict(dts_very_close, gts), accS_very_close)
    assert np.allclose(eval.compute_accuracy_strict(dts_close, gts), accS_close)
    assert np.allclose(eval.compute_accuracy_strict(dts_zero, gts), accS_zero)


def test_accuracy_relax() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated relax accuracy matches the expected value.
    """
    accS_perfect = np.ones(10)
    accS_close = np.ones(10)
    accS_very_close = np.ones(10)
    accS_zero = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert np.allclose(eval.compute_accuracy_relax(dts_perfect, gts), accS_perfect)
    assert np.allclose(eval.compute_accuracy_relax(dts_very_close, gts), accS_very_close)
    assert np.allclose(eval.compute_accuracy_relax(dts_close, gts), accS_close)
    assert np.allclose(eval.compute_accuracy_relax(dts_zero, gts), accS_zero)


def test_angle_error() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated angle error matches the expected value.
    """
    gts_norm = np.sqrt((gts**2).sum(-1) + 0.1**2)
    dts_close_norm = np.sqrt((dts_close**2).sum(-1) + 0.1**2)
    dts_very_close_norm = np.sqrt((dts_very_close**2).sum(-1) + 0.1**2)
    dts_perfect_norm = np.sqrt((dts_perfect**2).sum(-1) + 0.1**2)
    dts_zero_norm = np.sqrt((dts_zero**2).sum(-1) + 0.1**2)

    gts_nz_value = np.array([0.0, 1e-3, 1e-3, 1e-3, 1e3, 1e3, 1e3, 1.0, 1.0, 1.0], dtype=np.float64)

    ae_perfect = np.arccos(
        np.clip(
            0.1 / dts_perfect_norm * 0.1 / gts_norm
            + (gts_nz_value / gts_norm) * ((gts_nz_value + 0.00) / dts_perfect_norm),
            -1.0,
            1.0,
        )
    )
    ae_close = np.arccos(
        0.1 / dts_close_norm * 0.1 / gts_norm + (gts_nz_value / gts_norm) * ((gts_nz_value + 0.05) / dts_close_norm)
    )
    ae_very_close = np.arccos(
        0.1 / dts_very_close_norm * 0.1 / gts_norm
        + (gts_nz_value / gts_norm) * ((gts_nz_value + 0.01) / dts_very_close_norm)
    )
    ae_zero = np.arccos(0.1 / dts_zero_norm * 0.1 / gts_norm)

    assert np.allclose(eval.compute_angle_error(dts_perfect, gts), ae_perfect)
    assert np.allclose(eval.compute_angle_error(dts_very_close, gts), ae_very_close)
    assert np.allclose(eval.compute_angle_error(dts_close, gts), ae_close)
    assert np.allclose(eval.compute_angle_error(dts_zero, gts), ae_zero)


def test_true_positives() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated true positive count the expected value.
    """
    assert eval.compute_true_positives(dts_dynamic, gts_dynamic) == 2


def test_true_negatives() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated true negative count the expected value.
    """
    assert eval.compute_true_negatives(dts_dynamic, gts_dynamic) == 3


def test_false_positives() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated false positive count the expected value.
    """
    assert eval.compute_false_positives(dts_dynamic, gts_dynamic) == 3


def test_false_negatives() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated false negative count the expected value.
    """
    assert eval.compute_false_negatives(dts_dynamic, gts_dynamic) == 2


def test_metrics() -> None:
    """Initialize dummy ground truth labels and predictions.

    Verify that the metric breakdown has the correct subset counts and values.
    """
    metrics = eval.compute_metrics(
        dts_perfect, dts_dynamic, gts, gts_classes, gts_dynamic, gts_close, gts_valid, constants.FOREGROUND_BACKGROUND
    )

    for sub in metrics:
        if sub[0] == "Background" and sub[1] == "Dynamic":
            assert sub[3] == 0
        if sub[3] == 0:
            assert np.isnan(sub[5:8]).all()
        else:
            assert not np.isnan(sub[5:8]).any()

    counted_points: int = sum([int(sub[3]) for sub in metrics])
    valid_points: int = sum(gts_valid.astype(int))
    assert counted_points == valid_points

    tp: int = sum([int(sub[8]) for sub in metrics])
    assert tp == 2
    tn: int = sum([int(sub[9]) for sub in metrics])
    assert tp == 2  # The first TN is marked invalid
    fp: int = sum([int(sub[10]) for sub in metrics])
    assert fp == 3
    fn: int = sum([int(sub[11]) for sub in metrics])
    assert fn == 2

    nz_subsets = sum([1 for sub in metrics if not np.isnan(sub[4])])
    assert nz_subsets == 5

    assert np.allclose(sum([float(sub[4]) for sub in metrics if not np.isnan(sub[4])]) / nz_subsets, 0.0)
    assert np.allclose(sum([float(sub[5]) for sub in metrics if not np.isnan(sub[4])]) / nz_subsets, 1.0)
    assert np.allclose(sum([float(sub[6]) for sub in metrics if not np.isnan(sub[4])]) / nz_subsets, 1.0)
    assert np.allclose(sum([float(sub[7]) for sub in metrics if not np.isnan(sub[4])]) / nz_subsets, 0.0)


def test_average_metrics() -> pd.DataFrame:
    """Initialize dummy ground truth labels and predictions.

    Verify that the weighted average metric breakdown has the correct subset counts and values.
    """
    with Path(tempfile.TemporaryDirectory().name) as test_dir:
        test_dir.mkdir()
        anno_dir = test_dir / "annotations"
        anno_dir.mkdir()

        pred_dir = test_dir / "predictions"
        pred_dir.mkdir()

        ts = 111111111111111111

        write_annotation(gts_classes, gts_close, gts_dynamic, gts_valid, gts, ("log", ts), anno_dir)
        write_annotation(gts_classes, gts_close, gts_dynamic, gts_valid, gts, ("log_missing", ts), anno_dir)
        write_output_file(dts_perfect, dts_dynamic, ("log", ts), pred_dir)

        results_df = eval.evaluate_directories(anno_dir, pred_dir)

        assert len(results_df) == 8
        assert results_df.Count.sum() == 9

        assert np.allclose(results_df.EPE.mean(), 0.0)
        assert np.allclose(results_df["Accuracy Strict"].mean(), 1.0)
        assert np.allclose(results_df["Accuracy Relax"].mean(), 1.0)
        assert np.allclose(results_df["Angle Error"].mean(), 0.0)

        assert results_df.TP.sum() == 2
        assert results_df.TN.sum() == 2  # First true negative marked invalid
        assert results_df.FP.sum() == 3
        assert results_df.FN.sum() == 2

        assert results_df.groupby(["Class", "Motion"]).Count.sum().Background.Dynamic == 0

    results_dict = eval.results_to_dict(results_df)

    assert len(results_dict) == 38
    assert np.isnan([results_dict[k] for k in results_dict if k.endswith("Foreground/Static/Far")]).all()
    assert len([True for k in results_dict if "Background/Dyamic" in k]) == 0
    assert results_dict["Dynamic IoU"] == 2 / (3 + 2 + 2)
    assert results_dict["EPE 3-Way Average"] == 0.0


def test_eval_masks() -> pd.DataFrame:
    """Load an example mask a compare a susbset of it to some known values."""
    uuid = ("0c6e62d7-bdfa-3061-8d3d-03b13aa21f68", 315971436059707000)
    mask = get_eval_point_mask(uuid).numpy()

    inds = np.array([85029, 38229, 13164, 24083, 55903, 95251, 85976, 24794, 67262, 9531])
    gt_mask = np.array([False, False, True, True, True, True, False, True, True, True])
    assert np.all(mask[inds] == gt_mask)
