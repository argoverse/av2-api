"""Example of Rust-backed, PyTorch data-loader."""

import logging
from pathlib import Path
from typing import Final

from kornia.geometry.linalg import transform_points
from tqdm import tqdm

from av2.torch.data_loaders.detection import DetectionDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOME_DIR: Final = Path.home()


def main(
    root_dir: Path = HOME_DIR / "data" / "datasets",
    dataset_name: str = "av2",
    split_name: str = "val",
    num_accumulated_sweeps: int = 1,
    max_iterations: int = 10000,
) -> None:
    """Iterate over the detection data-loader.

    Dataset should live at ~/data/datasets/{dataset_name}/{split_name}.

    Args:
        root_dir: Root directory to the datasets.
        dataset_name: Name of the dataset (e.g., "av2").
        split_name: Name of the split (e.g., "val").
        num_accumulated_sweeps: Number of sweeps to accumulate.
        max_iterations: Maximum number of iterations for the data-loader example.
    """
    logger.info("Starting detection data-loader example ...")
    data_loader = DetectionDataLoader(
        root_dir,
        dataset_name,
        split_name,
        num_accumulated_sweeps=num_accumulated_sweeps,
    )
    for i, sweep in enumerate(tqdm(data_loader)):
        # 4x4 matrix representing the SE(3) transformation to city from ego-vehicle coordinates.
        city_SE3_ego_mat4 = sweep.city_SE3_ego.matrix()

        # Lidar (x,y,z) in meters and intensity (i).
        lidar_tensor = sweep.lidar.as_tensor()

        # Synchronized ring cameras.
        synchronized_images = data_loader.get_synchronized_images(i)

        # Transform the points to city coordinates.
        lidar_xyz_city = transform_points(city_SE3_ego_mat4, lidar_tensor[:, :3])

        # Cuboids might not be available (e.g., using the "test" split).
        if sweep.cuboids is not None:
            # Annotations in (x,y,z,l,w,h,yaw) format.
            # 1-DOF rotation.
            xyzlwh_t = sweep.cuboids.as_tensor()

            # Access cuboid category.
            category = sweep.cuboids.category

            # Access track uuid.
            track_uuid = sweep.cuboids.track_uuid

            # print(lidar_xyz_city, xyzlwh_t, category, track_uuid)

        if i >= max_iterations:
            logger.info(f"Reached max iterations of {max_iterations}!")
            break

    logger.info("Example complete!")


if __name__ == "__main__":
    main()
