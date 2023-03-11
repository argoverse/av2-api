"""Example of rust-backed, torch dataloader."""

from pathlib import Path

from av2.torch.dataloaders.sensor import Dataloader


def main():
    """Iterate over the dataloader."""

    # Dataset should live at ~/data/datasets/{dataset_name}/{split_name}
    root_dir = Path.home() / "data" / "datasets"
    split_name = "val"
    dataset_name = "av2"

    dataloader = Dataloader(root_dir, dataset_name, split_name)
    for lidar in dataloader:
        print(lidar)


if __name__ == "__main__":
    main()
