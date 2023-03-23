"""Scene Flow evaluation unit tests."""

import tempfile
from pathlib import Path

import numpy as np

import av2.evaluation.scene_flow.eval as eval
from av2.evaluation.scene_flow.example_submission import example_submission
from av2.evaluation.scene_flow.make_annotation_files import make_annotation_files
from av2.evaluation.scene_flow.make_mask_files import make_mask_files
from av2.evaluation.scene_flow.make_submission_archive import make_submission_archive

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent


def test_submission() -> None:
    """Test the whole pipeline for creating masks, annotations and submission files."""
    with Path(tempfile.TemporaryDirectory().name) as test_dir:
        test_dir.mkdir()
        mask_file = test_dir / "masks.zip"
        make_mask_files(str(mask_file), str(_TEST_DATA_ROOT), "test_data", "val")

        annotations_dir = test_dir / "annotations"
        make_annotation_files(str(annotations_dir), str(mask_file), str(_TEST_DATA_ROOT), "test_data", "val")

        predictions_dir = test_dir / "output"
        example_submission(str(predictions_dir), str(mask_file), str(_TEST_DATA_ROOT), "test_data")

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
        make_submission_archive(str(predictions_dir), str(mask_file), str(output_file))
