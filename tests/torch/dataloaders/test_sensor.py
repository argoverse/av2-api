"""Pytorch dataloader tests for the AV2 sensor dataset."""

from pathlib import Path
from typing import Final

from av2.torch.dataloaders.sensor import Av2
import torch.multiprocessing as mp

mp.set_start_method("fork")

TEST_DATA_DIR: Final[Path] = Path(".").resolve() / "test_data"


def test_av2_sensor_dataloader():
    """Test the av2 sensor dataloader."""
    dataloader = Av2(
        dataset_dir=TEST_DATA_DIR,
        split_name="",
    )
    for datum in dataloader:
        annotations = datum.annotations.as_tensor()
        lidar = datum.lidar.as_tensor()

        print(annotations)
        print(lidar)


if __name__ == "__main__":
    test_av2_sensor_dataloader()
