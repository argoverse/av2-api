"""Pytorch dataloader tests for the AV2 sensor dataset."""

from pathlib import Path
from typing import Final

TEST_DATA_DIR: Final[Path] = Path(__file__).parent.resolve() / "test_data"


def test_av2_sensor_dataloader() -> None:
    """Test the av2 sensor dataloader."""
    pass


if __name__ == "__main__":
    test_av2_sensor_dataloader()
