from pathlib import Path

import vedo

from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.rendering.graphics import cuboids, plotter, points

vedo.settings.allowInteraction = True


def render_scene() -> None:
    dataset_dir = Path.home() / "data" / "datasets" / "av2" / "sensor"
    dataloader = SensorDataloader(dataset_dir=dataset_dir)

    plot = plotter()
    for datum in dataloader:
        plot += points(datum.sweep)
        plot += cuboids(datum.annotations)
        plot.show(camera={"pos": [-60, 0, 40], "viewup": [1, 0, 0]})
        plot.clear()


if __name__ == "__main__":
    render_scene()
