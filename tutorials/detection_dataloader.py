"""Example of rust-backed, torch dataloader."""

from pathlib import Path

from kornia.geometry.linalg import transform_points
from tqdm import tqdm

from av2.torch.dataloaders.detection import DetectionDataloader


def main() -> None:
    """Iterate over the dataloader."""
    # Dataset should live at ~/data/datasets/{dataset_name}/{split_name}
    root_dir = Path.home() / "data" / "datasets"
    dataset_name = "av2"
    split_name = "val"
    num_accumulated_sweeps = 1

    dataloader = DetectionDataloader(root_dir, dataset_name, split_name, num_accumulated_sweeps=num_accumulated_sweeps)
    for sweep in tqdm(dataloader):
        city_SE3_ego_4x4 = sweep.city_SE3_ego.matrix()
        lidar_xyz_city = transform_points(city_SE3_ego_4x4, sweep.lidar_xyzi[:, :3])


if __name__ == "__main__":
    main()
