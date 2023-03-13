"""Example of rust-backed, torch dataloader."""

from pathlib import Path

from tqdm import tqdm

from av2.torch.dataloaders.scene_flow import SceneFlowDataloader
from av2.torch.dataloaders.sensor import Dataloader


def main():
    """Iterate over the dataloader."""

    # Dataset should live at ~/data/datasets/{dataset_name}/{split_name}
    root_dir = Path.home() / "data" / "datasets"
    dataset_name = "av2"
    split_name = "val"
    num_accum_sweeps = 5

    dataloader = SceneFlowDataloader(root_dir, dataset_name, split_name, num_accum_sweeps=num_accum_sweeps)
    for sweep, next_sweep in tqdm(dataloader):
        print(sweep, next_sweep)


if __name__ == "__main__":
    main()
