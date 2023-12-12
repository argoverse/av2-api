"""Scene Flow evaluation unit tests."""

import tempfile
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import numpy as np
import pandas as pd

import av2.evaluation.scene_flow.eval as eval
from av2.evaluation.scene_flow.example_submission import example_submission
from av2.evaluation.scene_flow.make_annotation_files import make_annotation_files
from av2.evaluation.scene_flow.make_mask_files import make_mask_files
from av2.evaluation.scene_flow.make_submission_archive import (
    make_submission_archive,
    validate,
)
from av2.evaluation.scene_flow.utils import compute_eval_point_mask
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader

_TEST_DATA_ROOT: Final = Path(__file__).resolve().parent.parent.parent


def _zipdir(directory: Path, output_file: Path) -> None:
    """Zip a directory into output_file.

    Args:
        directory: The directory to recursively zip up.
        output_file: The name of the output archive.
    """
    with ZipFile(output_file, "w") as zf:
        for f in directory.rglob("**"):
            zf.write(f, arcname=str(f.relative_to(directory)))


def test_submission() -> None:
    """Test the whole pipeline for creating masks, annotations and submission files."""
    test_dir = Path(tempfile.TemporaryDirectory().name)
    test_dir.mkdir()
    mask_file = test_dir / "masks.zip"
    make_mask_files(str(mask_file), str(_TEST_DATA_ROOT), "test_data", "val")

    annotations_dir = test_dir / "annotations"
    make_annotation_files(
        str(annotations_dir),
        str(mask_file),
        str(_TEST_DATA_ROOT),
        "test_data",
        "val",
    )

    predictions_dir = test_dir / "output"
    example_submission(
        str(predictions_dir), str(mask_file), str(_TEST_DATA_ROOT), "test_data"
    )

    results = eval.evaluate(str(annotations_dir), str(predictions_dir))
    for metric in results:
        if metric.startswith("Accuracy"):
            if "Static" in metric:
                assert np.allclose(results[metric], 1.0)
            elif "Dynamic" in metric and "Strict" in metric:
                assert np.isnan(results[metric]) or np.allclose(results[metric], 0.0)
        elif metric.startswith("EPE"):
            if "Static" in metric:
                if "Background" in metric:
                    assert np.allclose(results[metric], 0.0)
                elif "Foreground" in metric:
                    assert results[metric] < 0.06
        elif metric.startswith("Angle"):
            if "Static" in metric and "Background" in metric:
                assert results[metric] < 1e-4

    output_file = test_dir / "submission.zip"
    success = make_submission_archive(
        str(predictions_dir), str(mask_file), str(output_file)
    )
    assert success
    assert output_file.stat().st_size > 0

    annotation_files = list(annotations_dir.rglob("*.feather"))
    print(
        [
            anno_file.relative_to(annotations_dir).as_posix()
            for anno_file in annotation_files
        ]
    )
    with ZipFile(output_file, "r") as zf:
        files = {f.filename for f in zf.filelist}
    print(files)

    results_zip = eval.results_to_dict(eval.evaluate_zip(annotations_dir, output_file))
    for metric in results:
        assert np.allclose(results[metric], results_zip[metric], equal_nan=True)

    empty_predictions_dir = test_dir / "bad_output_1"
    empty_predictions_dir.mkdir()
    success = make_submission_archive(
        str(empty_predictions_dir), str(mask_file), str(output_file)
    )
    assert not success

    failed = False
    try:
        validate(empty_predictions_dir, mask_file)
    except FileNotFoundError:
        failed = True
    assert failed

    # Missing a column
    log_id = "7fab2350-7eaf-3b7e-a39d-6937a4c1bede"
    timestamp_ns = 315966265259836000
    data_loader = SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "val")
    sweep_0, sweep_1, s1_SE3_s0, _ = data_loader[0]
    mask = compute_eval_point_mask((sweep_0, sweep_1, s1_SE3_s0, None))
    npts = mask.sum()
    bad_df = pd.DataFrame(
        {
            "flow_tx_m": np.zeros(npts, dtype=np.float16),
            "flow_ty_m": np.zeros(npts, dtype=np.float16),
            "flow_tz_m": np.zeros(npts, dtype=np.float16),
        }
    )
    bad_cols_predictions_dir = test_dir / "bad_output_2" / log_id
    bad_cols_predictions_dir.mkdir(parents=True, exist_ok=True)
    bad_df.to_feather(bad_cols_predictions_dir / f"{timestamp_ns}.feather")
    failed = False
    try:
        validate(bad_cols_predictions_dir.parent, mask_file)
    except ValueError as e:
        print(e)
        assert "contain is_dynamic" in str(e)
        failed = True
    assert failed

    # Wrong dynamic column type
    bad_df = pd.DataFrame(
        {
            "flow_tx_m": np.zeros(npts, dtype=np.float16),
            "flow_ty_m": np.zeros(npts, dtype=np.float16),
            "flow_tz_m": np.zeros(npts, dtype=np.float16),
            "is_dynamic": np.zeros(npts, dtype=np.float16),
        }
    )
    bad_type_predictions_dir = test_dir / "bad_output_3" / log_id
    bad_type_predictions_dir.mkdir(parents=True, exist_ok=True)
    bad_df.to_feather(bad_type_predictions_dir / f"{timestamp_ns}.feather")
    failed = False
    try:
        validate(bad_type_predictions_dir.parent, mask_file)
    except ValueError as e:
        assert "column is_dynamic" in str(e)
        failed = True
    assert failed

    # Wrong flow column type
    bad_df = pd.DataFrame(
        {
            "flow_tx_m": np.zeros(npts, dtype=np.float16),
            "flow_ty_m": np.zeros(npts, dtype=np.float16),
            "flow_tz_m": np.zeros(npts, dtype=np.float32),
            "is_dynamic": np.zeros(npts, dtype=bool),
        }
    )
    bad_type_2_predictions_dir = test_dir / "bad_output_4" / log_id
    bad_type_2_predictions_dir.mkdir(exist_ok=True, parents=True)
    bad_df.to_feather(bad_type_2_predictions_dir / f"{timestamp_ns}.feather")
    failed = False
    try:
        validate(bad_type_2_predictions_dir.parent, mask_file)
    except ValueError as e:
        assert "column flow_tz_m" in str(e)
        failed = True
    assert failed

    # extra column
    bad_df = pd.DataFrame(
        {
            "flow_tx_m": np.zeros(npts, dtype=np.float16),
            "flow_ty_m": np.zeros(npts, dtype=np.float16),
            "flow_tz_m": np.zeros(npts, dtype=np.float16),
            "is_dynamic": np.zeros(npts, dtype=bool),
            "is_static": np.zeros(npts, dtype=bool),
        }
    )
    extra_col_predictions_dir = test_dir / "bad_output_5" / log_id
    extra_col_predictions_dir.mkdir(parents=True, exist_ok=True)
    bad_df.to_feather(extra_col_predictions_dir / f"{timestamp_ns}.feather")
    failed = False
    try:
        validate(extra_col_predictions_dir.parent, mask_file)
    except ValueError as e:
        assert "extra" in str(e)
        failed = True
    assert failed

    # wrong length
    bad_df = pd.DataFrame(
        {
            "flow_tx_m": np.zeros(npts + 1, dtype=np.float16),
            "flow_ty_m": np.zeros(npts + 1, dtype=np.float16),
            "flow_tz_m": np.zeros(npts + 1, dtype=np.float16),
            "is_dynamic": np.zeros(npts + 1, dtype=bool),
        }
    )
    wrong_len_predictions_dir = test_dir / "bad_output_6" / log_id
    wrong_len_predictions_dir.mkdir(exist_ok=True, parents=True)
    bad_df.to_feather(wrong_len_predictions_dir / f"{timestamp_ns}.feather")
    failed = False
    try:
        validate(wrong_len_predictions_dir.parent, mask_file)
    except ValueError as e:
        assert "rows" in str(e)
        failed = True
    assert failed
