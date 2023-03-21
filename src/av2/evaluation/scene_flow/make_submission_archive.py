"""Validate and package a set of prediction files for submission to the leaderboard."""

import argparse
import json
from pathlib import Path
from typing import Dict, Final
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rich.progress import track

import av2.evaluation.scene_flow.utils

SUBMISSION_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m", "is_dynamic")


def validate(root_dir: Path, fmt: Dict[str, int]) -> None:
    """Validate the filenames and shapes of all predictions required for submission.

    Args:
        root_dir: Path to the top level submission file directory.
        fmt: Dictionary containing all the files needed for submission and the number of points in that file.

    Raises:
        FileNotFoundError: If any of the required files are missing
        ValueError: If any supplied file is malformed
    """
    for filename in track(fmt.keys(), description="Validating..."):
        input_file = submission_root / filename
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


def zip(root_dir: Path, fmt: Dict[str, int], output_file: Path) -> None:
    """Package all validated submission files into a zip archive.

    Args:
        root_dir: Path to the top level submission file directory.
        fmt: Dictionary containing all the files needed for submission and the number of points in that file.
        output_file: File to store the zip archive in.
    """
    with ZipFile(output_file, "w") as myzip:
        for filename in track(fmt.keys(), description="Zipping..."):
            input_file = submission_root / filename
            myzip.write(input_file, arcname=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "make_submission_archive",
        description="Validate a set of submission files and then "
        "package them into a zip file. Files should be in the format "
        "specified in SUBMISSION_FORMAT.md",
    )

    parser.add_argument("submission_root", type=str, help="location of the submission files")
    parser.add_argument("--output_file", type=str, default="submission.zip", help="name of output archive file")
    args = parser.parse_args()

    format_path = Path(av2.evaluation.scene_flow.utils.__file__).parent / "test_submission_format.json"
    with open(format_path, "r") as f:
        fmt = json.load(f)

    submission_root = Path(args.submission_root)
    output_file = Path(args.output_file)
    try:
        validate(submission_root, fmt)
    except Exception as e:
        print("Input validation failed with:")
        print(e)

    zip(submission_root, fmt, output_file)
