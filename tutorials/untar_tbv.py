# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Script to untar tar.gz files from the Trust but Verify (TbV) dataset."""

import subprocess
from pathlib import Path
from typing import Final, Optional, Tuple

import click
from joblib import Parallel, delayed

NUM_TBV_SHARDS: Final[int] = 21


def run_command(
    cmd: str, return_output: bool = False
) -> Tuple[Optional[bytes], Optional[bytes]]:
    """Execute a system call, and block until the system call completes.

    Args:
        cmd: string, representing shell command
        return_output: whether to return STDOUT and STDERR data.

    Returns:
        Tuple of (stdout, stderr) output if return_output is True, else None
    """
    print(cmd)
    (stdout_data, stderr_data) = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE
    ).communicate()

    if return_output:
        return stdout_data, stderr_data
    return None, None


def untar_tbv_dataset(
    num_workers: int, shard_dirpath: Path, desired_tbv_dataroot: Path
) -> None:
    """Untar each of the tar.gz archives.

    Args:
        num_workers: Number of worker processes to use for simultaneous extraction
        shard_dirpath: Path to local directory where shard tar.gz files where downloaded.
        desired_tbv_dataroot: Path to local directory, where TbV logs will be extracted.

    Raises:
        RuntimeError: If one of the expected tar.gz shard files is missing.
    """
    desired_tbv_dataroot.mkdir(exist_ok=True, parents=True)

    jobs = []
    for i in range(NUM_TBV_SHARDS):
        shard_fpath = shard_dirpath / f"TbV_v1.0_shard{i}.tar.gz"
        if not shard_fpath.exists():
            raise RuntimeError(f"Shard file not found {shard_fpath}")
        cmd = f"tar -xvzf {shard_fpath} --directory {desired_tbv_dataroot}"
        jobs.append(cmd)

    if num_workers > 1:
        Parallel(n_jobs=num_workers)(delayed(run_command)(cmd) for cmd in jobs)
    else:
        for cmd in jobs:
            run_command(cmd)


@click.command(
    help="Extract TbV tar.gz archives that were previously downloaded to a local disk."
)
@click.option(
    "--num-workers",
    required=True,
    help="Number of worker processes to use for simultaneous extraction.",
    type=int,
)
@click.option(
    "--shard-dirpath",
    required=True,
    help="Path to local directory where shard tar.gz files were previously downloaded.",
    type=click.Path(exists=True),
)
@click.option(
    "--desired-tbv-dataroot",
    required=True,
    help="Path to local directory, where TbV logs will be extracted.",
    type=str,
)
def run_untar_tbv_dataset(
    num_workers: int, shard_dirpath: str, desired_tbv_dataroot: str
) -> None:
    """Click entry point for TbV tar.gz file extraction."""
    untar_tbv_dataset(
        num_workers=num_workers,
        shard_dirpath=Path(shard_dirpath),
        desired_tbv_dataroot=Path(desired_tbv_dataroot),
    )


if __name__ == "__main__":
    run_untar_tbv_dataset()
