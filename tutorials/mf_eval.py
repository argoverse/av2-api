"""Motion forecasting evaluation script for Argoverse 2 challenge submissions."""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Any, Dict, Final, List, NamedTuple, Tuple, cast

import click
from tqdm import tqdm

from av2.datasets.motion_forecasting.eval import metrics
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
from av2.utils.typing import NDArrayFloat

# Type alias for track predictions: (forecasted_trajectories, forecast_probabilities)
TrackPredictions = Tuple[NDArrayFloat, NDArrayFloat]

ResultsRow = Tuple[str, str, float, float, float, float, float, float, float, float]
RESULTS_COL_NAMES: Final = [
    "scenario_id",
    "track_id",
    "min_fde_k6",
    "min_fde_k1",
    "min_ade_k6",
    "min_ade_k1",
    "is_missed_k6",
    "is_missed_k1",
    "min_brier_fde_k6",
    "min_brier_ade_k6",
]


class TrackAnnotations(NamedTuple):
    """Ground truth annotations for a single track.
    
    Attributes:
        scenario_id: Unique identifier for the scenario.
        track_id: Unique identifier for the track within the scenario.
        is_focal_track: Whether this is a focal track to be evaluated.
        gt_trajectory_x: Ground truth x-coordinates of the trajectory.
        gt_trajectory_y: Ground truth y-coordinates of the trajectory.
    """
    scenario_id: str
    track_id: str
    is_focal_track: bool
    gt_trajectory_x: NDArrayFloat
    gt_trajectory_y: NDArrayFloat


def evaluate(test_annotation_file: Path, user_submission_file: str) -> Dict[str, Any]:
    """Evaluates the submission for a particular challenge phase and returns score.

    Args:
        test_annotation_file: Path to test_annotation_file on the server, for argoverse challenge this is path to the dataset split folder
        user_submission_file: Path to file submitted by the user, for argoverse challenge this is a parquet file
    """
    evaluation_start_time = time.time()

    if not user_submission_file.endswith('.parquet'):
        raise ValueError("Submission file was not in parquet format.")

    print("Starting Evaluation...")

    # Load annotations from parquet file
    annotation_df = pd.read_parquet(test_annotation_file)

    # Deserialize submission from parquet file
    print("Deserializing submission...")
    submission_load_start_time = time.time()
    submission = ChallengeSubmission.from_parquet(Path(user_submission_file))
    print(f"Deserialization complete, time elapsed: {time.time() - submission_load_start_time:.2f}s")

    # Score each track in the annotation file
    print("Scoring tracks...")
    track_scoring_start_time = time.time()
    results_rows: List[ResultsRow] = []
    for row in tqdm(annotation_df.itertuples()):
        if not row.is_focal_track:
            continue
        scenario_id = str(row.scenario_id)
        track_id = str(row.track_id)
        
        # Create properly typed TrackAnnotations
        track_annotations = TrackAnnotations(
            scenario_id=scenario_id,
            track_id=track_id,
            is_focal_track=bool(row.is_focal_track),
            gt_trajectory_x=cast(NDArrayFloat, row.gt_trajectory_x),
            gt_trajectory_y=cast(NDArrayFloat, row.gt_trajectory_y)
        )
        
        track_predictions = cast(TrackPredictions, submission.predictions[scenario_id][track_id])  # type: ignore[index]
        results_rows.append(compute_track_results(track_predictions, track_annotations))

    results_df = pd.DataFrame(results_rows, columns=RESULTS_COL_NAMES)
    print(f"Scoring complete, time elapsed: {time.time() - track_scoring_start_time:.2f}s")

    # Compute summary scores for submission
    summary_scores = compute_summary_scores(results_df)
    print(f"Evaluation job complete, total time elapsed: {time.time() - evaluation_start_time:.2f}s")

    return summary_scores


def compute_track_results(track_predictions: TrackPredictions, track_annotations: TrackAnnotations) -> ResultsRow:
    """Compute evaluation metrics for a single track.
    
    Args:
        track_predictions: Tuple of (forecasted_trajectories, forecast_probabilities) for the track.
        track_annotations: Ground truth annotations for the track.
        
    Returns:
        ResultsRow containing scenario_id, track_id, and all computed metrics.
    """
    annotation_trajectory = np.stack([track_annotations.gt_trajectory_x, track_annotations.gt_trajectory_y], axis=-1)
    forecast_probabilities = track_predictions[1]
    forecasted_trajectories = track_predictions[0]

    # Compute minFDE, minADE, and miss rate
    fde = metrics.compute_fde(forecasted_trajectories, annotation_trajectory)
    min_fde_k1 = fde[0]
    min_fde_k6_idx = np.argmin(fde[:6])
    min_fde_k6 = fde[min_fde_k6_idx]

    is_missed = metrics.compute_is_missed_prediction(forecasted_trajectories, annotation_trajectory)
    is_missed_k1 = is_missed[0]
    is_missed_k6 = is_missed[min_fde_k6_idx]

    ade = metrics.compute_ade(forecasted_trajectories, annotation_trajectory)
    min_ade_k1 = ade[0]
    min_ade_k6_idx = np.argmin(ade[:6])
    min_ade_k6 = ade[min_ade_k6_idx]

    # Compute probabilistic metrics using b-minFDE and b-minADE
    brier_fde = metrics.compute_brier_fde(forecasted_trajectories,
                                          annotation_trajectory,
                                          forecast_probabilities,
                                          normalize=True)
    min_brier_fde_k6 = brier_fde[min_fde_k6_idx]

    brier_ade = metrics.compute_brier_ade(forecasted_trajectories,
                                          annotation_trajectory,
                                          forecast_probabilities,
                                          normalize=True)
    min_brier_ade_k6 = brier_ade[min_ade_k6_idx]

    # Build corresponding row for results dataframe
    track_results: ResultsRow = (
        track_annotations.scenario_id,
        track_annotations.track_id,
        float(min_fde_k6),
        float(min_fde_k1),
        float(min_ade_k6),
        float(min_ade_k1),
        float(is_missed_k6),
        float(is_missed_k1),
        float(min_brier_fde_k6),
        float(min_brier_ade_k6),
    )
    return track_results


def compute_summary_scores(results_df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics across all tracks.
    
    Args:
        results_df: DataFrame containing per-track evaluation results.
        
    Returns:
        Dictionary mapping metric names to their aggregated scores.
    """
    avg_min_fde_k6 = np.around(results_df["min_fde_k6"].mean(), decimals=4)
    avg_min_fde_k1 = np.around(results_df["min_fde_k1"].mean(), decimals=4)
    avg_min_ade_k6 = np.around(results_df["min_ade_k6"].mean(), decimals=4)
    avg_min_ade_k1 = np.around(results_df["min_ade_k1"].mean(), decimals=4)
    miss_rate_k6 = np.around(results_df["is_missed_k6"].mean(), decimals=4)
    miss_rate_k1 = np.around(results_df["is_missed_k1"].mean(), decimals=4)
    avg_min_brier_fde_k6 = np.around(results_df["min_brier_fde_k6"].mean(), decimals=4)
    avg_min_brier_ade_k6 = np.around(results_df["min_brier_ade_k6"].mean(), decimals=4)

    summary_scores = {
        "minFDE (K=6)": avg_min_fde_k6,
        "minFDE (K=1)": avg_min_fde_k1,
        "minADE (K=6)": avg_min_ade_k6,
        "minADE (K=1)": avg_min_ade_k1,
        "MR (K=6)": miss_rate_k6,
        "MR (K=1)": miss_rate_k1,
        "brier-minFDE (K=6)": avg_min_brier_fde_k6,
        "brier-minADE (K=6)": avg_min_brier_ade_k6,
    }
    return summary_scores


@click.command()
@click.option(
    "--test-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to test annotations parquet file."
)
@click.option(
    "--submission-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to submission parquet file."
)
def main(test_file: Path, submission_file: Path) -> None:
    """Evaluate a submission file against test annotations."""
    result_dict = evaluate(test_file, str(submission_file))
    
    print("\nFinal Results:")
    for metric, score in result_dict.items():
        print(f"{metric}: {score}")


if __name__ == "__main__":
    main()