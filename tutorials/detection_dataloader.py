"""Example of rust-backed, torch dataloader."""

from pathlib import Path

from kornia.geometry.linalg import transform_points
from tqdm import tqdm

from av2.torch.dataloaders.detection import DetectionDataloader


def main(
    root_dir: Path = Path.home() / "data" / "datasets",
    dataset_name: str = "av2",
    split_name: str = "val",
    num_accumulated_sweeps: int = 1,
) -> None:
    """Iterate over the detection dataloader.

    Dataset should live at ~/data/datasets/{dataset_name}/{split_name}.

    Args:
        root_dir: Root directory to the datasets.
        dataset_name: Name of the dataset (e.g., "av2").
        split_name: Name of the split (e.g., "val").
        num_accumulated_sweeps: Number of sweeps to accumulate.
    """

    dataloader = DetectionDataloader(root_dir, dataset_name, split_name, num_accumulated_sweeps=num_accumulated_sweeps)
    for sweep in tqdm(dataloader):
        # 4x4 matrix representing the SE(3) transformation to city from ego-vehicle coordinates.
        city_SE3_ego_4x4 = sweep.city_SE3_ego.matrix()

        # Lidar (x,y,z) in meters and intensity (i).
        lidar_xyzi_ego = sweep.lidar_xyzi[:, :3]

        # Transform the points to city coordinates.
        lidar_xyz_city = transform_points(city_SE3_ego_4x4, lidar_xyzi_ego[:, :3])


if __name__ == "__main__":
    main()
