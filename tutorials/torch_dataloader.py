"""Example of rust-backed, torch dataloader."""

from pathlib import Path

from tqdm import tqdm

from av2.torch.dataloaders.sensor import Dataloader


def main() -> None:
    """Iterate over the dataloader."""
    # Dataset should live at ~/data/datasets/{dataset_name}/{split_name}
    root_dir = Path.home() / "data" / "datasets"
    dataset_name = "av2"
    split_name = "val"
    num_accum_sweeps = 1

    dataloader = Dataloader(root_dir, dataset_name, split_name, num_accum_sweeps=num_accum_sweeps)
    for sweep in tqdm(dataloader):
        print(sweep.lidar.as_tensor())


if __name__ == "__main__":
    main()
