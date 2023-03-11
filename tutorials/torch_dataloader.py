from pathlib import Path

from av2.torch.dataloaders.sensor import Dataloader


def main():
    """Example of rust-backed, torch dataloader."""
    root_dir = Path.home() / "data" / "datasets"
    split_name = "val"
    dataset_name = "av2"
    # num_accum_sweeps = 1

    dataloader = Dataloader(root_dir, dataset_name, split_name)
    for lidar in dataloader:
        print(lidar)


if __name__ == "__main__":
    main()
