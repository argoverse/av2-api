"""Pytorch dataloader tests for the AV2 sensor dataset."""

from pathlib import Path
from typing import Final

import polars as pl
import torch.multiprocessing as mp

from av2.torch.dataloaders.sensor import Av2
from av2.torch.dataloaders.utils import read_feather
from av2.utils.io import read_city_SE3_ego

mp.set_start_method("fork")

TEST_DATA_DIR: Final[Path] = Path(".").resolve() / "test_data"


def test_av2_sensor_dataloader() -> None:
    """Test the av2 sensor dataloader."""
    dataloader = Av2(
        dataset_dir=TEST_DATA_DIR,
        split_name="",
    )

    datum = next(iter(dataloader))
    log_id, _ = datum.sweep_uuid

    src = TEST_DATA_DIR / log_id
    city_SE3_ego_mapping = read_city_SE3_ego(src)
    annotations_original = read_feather(src / "annotations.feather")

    annotations_original = annotations_original.sort(["track_uuid", "timestamp_ns"])
    partitions = annotations_original.partition_by("track_uuid", maintain_order=True, as_dict=True)
    for _, annotations in partitions.items():
        # for annotations in annotations_original.groupby(pl.col(["track_uuid"]), maintain_order=True):
        # timestamp_ns = annotations["timestamp_ns"][0]

        annotations_xyz_ego = annotations.select(pl.col(["tx_m", "ty_m", "tz_m"]))
        dataframe = pl.concat(
            [
                annotations.select(pl.col(["timestamp_ns", "track_uuid"])),
                annotations_xyz_ego,
            ],
            how="horizontal",
        )

        txyz_npy = dataframe.select(pl.col(["tx_m", "ty_m", "tz_m"])).to_numpy()
        timestamps_npy = dataframe.select(pl.col(["timestamp_ns"])).to_numpy()
        for i, _ in enumerate(txyz_npy):
            if i == 0:
                txyz_city_ti = city_SE3_ego_mapping[timestamps_npy[i, 0]].transform_point_cloud(txyz_npy[i][None])
                txyz_city_ti_plus_one = city_SE3_ego_mapping[timestamps_npy[i + 1, 0]].transform_point_cloud(
                    txyz_npy[i + 1][None]
                )

                diff = txyz_city_ti_plus_one - txyz_city_ti
                dt = (timestamps_npy[i + 1] - timestamps_npy[i]) * 1e-9
                vel = diff / dt

                vel_df = pl.from_numpy(vel, columns=["vx_m", "vy_m", "vz_m"])
                current_dataframe = pl.concat(
                    (dataframe.select(pl.col(["timestamp_ns", "track_uuid"]))[i], vel_df), how="horizontal"
                )

        test = datum.annotations.dataframe.join(current_dataframe, on=["track_uuid", "timestamp_ns"])
        if len(test) == 0:
            continue


if __name__ == "__main__":
    test_av2_sensor_dataloader()
