"""Validate and package a set of prediction files for submission to the leaderboard."""
import json
from pathlib import Path
from typing import Dict, Final
from zipfile import ZipFile

import click
import numpy as np
import pandas as pd
from rich.progress import track

import av2.evaluation.scene_flow.utils

SUBMISSION_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m", "is_dynamic")


def validate(submission_dir: Path, mask_file; Path) -> None:
    """Validate the filenames and shapes of all predictions required for submission.

    Args:
        submission_dir: Path to the top level submission file directory.
        mask_file: Archive containing all the mask files required for submission.

    Raises:
        FileNotFoundError: If any of the required files are missing
        ValueError: If any supplied file is malformed
    """
    with ZipFile(mask_file, "r") as masks:
        pass
    for filename in track(fmt.keys(), description="Validating..."):
        input_file = submission_dir / filename
        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} not found in submission directory")
        pred = pd.read_feather(input_file)

        for c in SUBMISSION_COLUMNS:
            if c not in pred.columns:
                raise ValueError(f"{input_file} does not contain {c}")
            if c == "is_dynamic":
                if pred[c].dtype != bool:
                    raise ValueError(f"{input_file} column {c} should be bool but is {pred[c].dtype}")
            else:
                if pred[c].dtype != np.float16:
                    raise ValueError(f"{input_file} column {c} should be float16 but is {pred[c].dtype}")

        if len(pred.columns) > 4:
            raise ValueError(f"{input_file} contains extra columns")

        if len(pred) != fmt[filename]:
            raise ValueError(f"{input_file} has {len(pred)} rows but it should have {fmt[filename]}")


def zip(submission_dir: Path, fmt: Dict[str, int], output_file: Path) -> None:
    """Package all validated submission files into a zip archive.

    Args:
        submission_dir: Path to the top level submission file directory.
        fmt: Dictionary containing all the files needed for submission and the number of points in that file.
        output_file: File to store the zip archive in.
    """
    with ZipFile(output_file, "w") as myzip:
        for filename in track(fmt.keys(), description="Zipping..."):
            input_file = submission_dir / filename
            myzip.write(input_file, arcname=filename)


@click.command()
@click.argument("submission_dir", type=str)
@click.option("--output_filename", type=str, help="name of the output archive file", default="submission.zip")
def make_submission_archive(submission_dir: str, output_filename: str) -> None:
    """Package prediction files into a zip archive for submission.

    Args:
        submission_dir: Directory containing the prediction files to submit.
        output_filename: Name of the submission archive.
    """
    format_path = Path(av2.evaluation.scene_flow.utils.__file__).parent / "test_submission_format.json"
    with open(format_path, "r") as f:
        fmt = json.load(f)

    output_file = Path(output_filename)
    try:
        validate(Path(submission_dir), fmt)
    except Exception as e:
        print("Input validation failed with:")
        print(e)

    zip(Path(submission_dir), fmt, output_file)


if __name__ == "__main__":
    make_submission_archive()
