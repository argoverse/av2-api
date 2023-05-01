"""Validate and package a set of prediction files for submission to the leaderboard."""
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import click
import numpy as np
import pandas as pd
from rich.progress import track

SUBMISSION_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m", "is_dynamic")


def validate(submission_dir: Path, mask_file: Path) -> None:
    """Validate the filenames and shapes of all predictions required for submission.

    Args:
        submission_dir: Path to the top level submission file directory.
        mask_file: Archive containing all the mask files required for submission.

    Raises:
        FileNotFoundError: If any of the required files are missing
        ValueError: If any supplied file is malformed
    """
    with ZipFile(mask_file, "r") as masks:
        mask_files = [
            f.filename for f in masks.filelist if f.filename.endswith(".feather")
        ]
        for filename in track(mask_files, description="Validating..."):
            input_file = submission_dir / filename
            if not input_file.exists():
                raise FileNotFoundError(
                    f"{input_file} not found in submission directory"
                )
            pred = pd.read_feather(input_file)
            expected_num_points = pd.read_feather(masks.open(filename)).sum().item()

            for c in SUBMISSION_COLUMNS:
                if c not in pred.columns:
                    raise ValueError(f"{input_file} does not contain {c}")
                if c == "is_dynamic":
                    if pred[c].dtype != bool:
                        raise ValueError(
                            f"{input_file} column {c} should be bool but is {pred[c].dtype}"
                        )
                else:
                    if pred[c].dtype != np.float16:
                        raise ValueError(
                            f"{input_file} column {c} should be float16 but is {pred[c].dtype}"
                        )

                if len(pred.columns) > 4:
                    raise ValueError(f"{input_file} contains extra columns")

                if len(pred) != expected_num_points:
                    raise ValueError(
                        f"{input_file} has {len(pred)} rows but it should have {expected_num_points}"
                    )


def zip(submission_dir: Path, mask_file: Path, output_file: Path) -> None:
    """Package all validated submission files into a zip archive.

    Args:
        submission_dir: Path to the top level submission file directory.
        mask_file: Archive containing all the mask files required for submission.
        output_file: File to store the zip archive in.
    """
    with ZipFile(mask_file, "r") as masks:
        mask_files = [
            f.filename for f in masks.filelist if f.filename.endswith(".feather")
        ]
    with ZipFile(output_file, "w") as myzip:
        for filename in track(mask_files, description="Zipping..."):
            input_file = submission_dir / filename
            myzip.write(input_file, arcname=filename)


def make_submission_archive(
    submission_dir: str, mask_file: str, output_filename: str
) -> bool:
    """Package prediction files into a zip archive for submission.

    Args:
        submission_dir: Directory containing the prediction files to submit.
        mask_file: Archive containing all the mask files required for submission.
        output_filename: Name of the submission archive.

    Returns:
        True if validation and zipping succeeded, False otherwise.
    """
    output_file = Path(output_filename)
    try:
        validate(Path(submission_dir), Path(mask_file))
    except (FileNotFoundError, ValueError) as e:
        print(f"Input validation failed with: {e}")
        return False

    zip(Path(submission_dir), Path(mask_file), output_file)
    return True


@click.command()
@click.argument("submission_dir", type=str)
@click.argument("mask_file", type=str)
@click.option(
    "--output_filename",
    type=str,
    help="name of the output archive file",
    default="submission.zip",
)
def _make_submission_archive_entry(
    submission_dir: str, mask_file: str, output_filename: str
) -> bool:
    return make_submission_archive(submission_dir, mask_file, output_filename)


if __name__ == "__main__":
    _make_submission_archive_entry()
