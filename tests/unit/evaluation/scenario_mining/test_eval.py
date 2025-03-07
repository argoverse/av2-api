"""Scenario mining evaluation unit tests."""

import matplotlib
from pathlib import Path
from typing import Final
import sys

from av2.evaluation.scenario_mining.eval import evaluate, load


matplotlib.use("Agg")
if sys.stdout is None:
    sys.stdout = open("stdout.log", "w")

TEST_DATA_DIR: Final[Path] = Path(__file__).parent.resolve() / "data"


def test_evaluate() -> None:
    """Test End-to-End Forecasting evaluation."""
    pred_pkl = TEST_DATA_DIR / "combined_predictions_dev.pkl"
    gt_pkl = TEST_DATA_DIR / "combined_gt_dev.pkl"

    objective_metric = "HOTA"
    max_range_m = 100
    dataset_dir = TEST_DATA_DIR
    out = str(TEST_DATA_DIR / "eval_results")

    predictions = load(pred_pkl)
    ground_truth = load(gt_pkl)

    evaluate(predictions, ground_truth, objective_metric, max_range_m, dataset_dir, out)


test_evaluate()
