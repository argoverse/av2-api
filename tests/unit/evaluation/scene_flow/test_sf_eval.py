"""Scene Flow evaluation unit tests."""

import tempfile
from pathlib import Path

import numpy as np

import av2.evaluation.scene_flow.constants as constants
import av2.evaluation.scene_flow.eval as eval
from av2.evaluation.scene_flow.make_annotation_files import write_annotation
from av2.evaluation.scene_flow.utils import write_output_file
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

gts_dynamic: NDArrayBool = np.array(
    [False, False, False, False, True, True, True, True, False, False]
)
gts_classes: NDArrayInt = np.array([0, 0, 0, 0, 17, 17, 1, 11, 0, 23])
gts_valid: NDArrayBool = np.array(
    [False, True, True, True, True, True, True, True, True, True]
)
gts_close: NDArrayBool = np.array(
    [True, True, False, False, True, True, False, False, True, True]
)

dts_dynamic: NDArrayBool = np.array(
    [False, True, False, True, False, True, False, True, False, True]
)


def test_end_point_error() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated end-point error matches the expected value.
    """
    epe_perfect = np.zeros(10)
    epe_very_close = (np.sqrt(0.01**2 * 3, dtype=np.float64) * np.ones(10)).astype(
        np.float64
    )
    epe_close = (np.sqrt(0.05**2 * 3, dtype=np.float64) * np.ones(10)).astype(
        np.float64
    )
    epe_zero = np.array(
        [0.0e00, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e3, 1.0e3, 1.0e3, 1.0e00, 1.0e00, 1.0e00],
        dtype=np.float64,
    )

    assert np.allclose(eval.compute_end_point_error(dts_perfect, gts), epe_perfect)
    assert np.allclose(
        eval.compute_end_point_error(dts_very_close, gts), epe_very_close
    )
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
    assert np.allclose(
        eval.compute_accuracy_strict(dts_very_close, gts), accS_very_close
    )
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
    assert np.allclose(
        eval.compute_accuracy_relax(dts_very_close, gts), accS_very_close
    )
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

    gts_nz_value = np.array(
        [0.0, 1e-3, 1e-3, 1e-3, 1e3, 1e3, 1e3, 1.0, 1.0, 1.0], dtype=np.float64
    )

    ae_perfect = np.arccos(
        np.clip(
            0.1 / dts_perfect_norm * 0.1 / gts_norm
            + (gts_nz_value / gts_norm) * ((gts_nz_value + 0.00) / dts_perfect_norm),
            -1.0,
            1.0,
        )
    )
    ae_close = np.arccos(
        0.1 / dts_close_norm * 0.1 / gts_norm
        + (gts_nz_value / gts_norm) * ((gts_nz_value + 0.05) / dts_close_norm)
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
        dts_perfect,
        dts_dynamic,
        gts,
        gts_classes,
        gts_dynamic,
        gts_close,
        gts_valid,
        constants.FOREGROUND_BACKGROUND_BREAKDOWN,
    )
    num_subsets = len(list(metrics.values())[0])
    assert num_subsets == 8

    for i in range(num_subsets):
        if metrics["Class"][i] == "Background" and metrics["Motion"][i] == "Dynamic":
            assert metrics["Count"][i] == 0
        if metrics["Count"][i] == 0:
            assert np.isnan(metrics["EPE"][i])
            assert np.isnan(metrics["ANGLE_ERROR"][i])
            assert np.isnan(metrics["ACCURACY_STRICT"][i])
            assert np.isnan(metrics["ACCURACY_RELAX"][i])
        else:
            assert not np.isnan(metrics["EPE"][i])
            assert not np.isnan(metrics["ANGLE_ERROR"][i])
            assert not np.isnan(metrics["ACCURACY_STRICT"][i])
            assert not np.isnan(metrics["ACCURACY_RELAX"][i])

    counted_points: int = sum(metrics["Count"])
    valid_points: int = sum(gts_valid.astype(int))
    assert counted_points == valid_points

    tp: int = sum(metrics["TP"])
    assert tp == 2  # The first TN is marked invalid
    fp: int = sum(metrics["FP"])
    assert fp == 3
    fn: int = sum(metrics["FN"])
    assert fn == 2

    nz_subsets = sum([1 for count in metrics["Count"] if count > 0])
    assert nz_subsets == 5

    assert np.allclose(
        sum(
            [
                accr
                for accr, count in zip(metrics["ACCURACY_RELAX"], metrics["Count"])
                if count > 0
            ]
        )
        / nz_subsets,
        1.0,
    )
    assert np.allclose(
        sum(
            [
                accs
                for accs, count in zip(metrics["ACCURACY_STRICT"], metrics["Count"])
                if count > 0
            ]
        )
        / nz_subsets,
        1.0,
    )
    assert np.allclose(
        sum(
            [
                ae
                for ae, count in zip(metrics["ANGLE_ERROR"], metrics["Count"])
                if count > 0
            ]
        )
        / nz_subsets,
        0.0,
    )
    assert np.allclose(
        sum([epe for epe, count in zip(metrics["EPE"], metrics["Count"]) if count > 0])
        / nz_subsets,
        0.0,
    )


def test_average_metrics() -> None:
    """Initialize dummy ground truth labels and predictions.

    Verify that the weighted average metric breakdown has the correct subset counts and values.
    """
    test_dir = Path(tempfile.TemporaryDirectory().name)
    test_dir.mkdir()
    anno_dir = test_dir / "annotations"
    anno_dir.mkdir()

    pred_dir = test_dir / "predictions"
    pred_dir.mkdir()

    timestamp_ns_1 = 111111111111111111
    timestamp_ns_2 = 111111111111111112

    write_annotation(
        gts_classes,
        gts_close,
        gts_dynamic,
        gts_valid,
        gts,
        ("log", timestamp_ns_1),
        anno_dir,
    )
    write_annotation(
        gts_classes,
        gts_close,
        gts_dynamic,
        gts_valid,
        gts,
        ("log", timestamp_ns_2),
        anno_dir,
    )

    write_annotation(
        gts_classes,
        gts_close,
        gts_dynamic,
        gts_valid,
        gts,
        ("log_missing", timestamp_ns_1),
        anno_dir,
    )

    write_output_file(dts_perfect, dts_dynamic, ("log", timestamp_ns_1), pred_dir)
    write_output_file(dts_perfect, dts_dynamic, ("log", timestamp_ns_2), pred_dir)

    results_df = eval.evaluate_directories(anno_dir, pred_dir)

    assert len(results_df) == 16
    assert results_df.Count.sum() == 18

    assert np.allclose(results_df.EPE.mean(), 0.0)
    assert np.allclose(results_df["ACCURACY_STRICT"].mean(), 1.0)
    assert np.allclose(results_df["ACCURACY_RELAX"].mean(), 1.0)
    assert np.allclose(results_df["ANGLE_ERROR"].mean(), 0.0)

    assert results_df.TP.sum() == 2 * 2
    assert results_df.TN.sum() == 2 * 2  # First true negative marked invalid
    assert results_df.FP.sum() == 3 * 2
    assert results_df.FN.sum() == 2 * 2

    assert results_df.groupby(["Class", "Motion"]).Count.sum().Background.Dynamic == 0
    results_dict = eval.results_to_dict(results_df)

    assert len(results_dict) == 38
    assert np.isnan(
        [results_dict[k] for k in results_dict if k.endswith("Foreground/Static/Far")]
    ).all()
    assert len([True for k in results_dict if "Background/Dyamic" in k]) == 0
    assert results_dict["Dynamic IoU"] == 2 / (3 + 2 + 2)
    assert results_dict["EPE 3-Way Average"] == 0.0
