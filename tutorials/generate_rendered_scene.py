# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Render a colorized point cloud from the sensor dataset."""

from pathlib import Path

import numpy as np
import vedo

from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.rendering.graphics import cuboids, egovehicle, lanes, plotter, point_cloud
from av2.utils.typing import NDArrayByte, NDArrayInt

vedo.settings.allowInteraction = True


def render_scene() -> None:
    """Render a scene from the sensor dataset."""
    dataset_dir = Path.home() / "data" / "datasets" / "av2" / "sensor"
    dataloader = SensorDataloader(dataset_dir=dataset_dir)

    plot = plotter()
    video = vedo.Video(
        fps=10,  # Lidar Hz
    )
    video.options = "-c:v libx264 -crf 24 -preset veryfast -pix_fmt yuv420p"
    for i, datum in enumerate(dataloader):
        plot += egovehicle()

        if datum.annotations is not None:
            plot += cuboids(datum.annotations)

        if datum.avm is not None:
            plot += lanes(datum.avm, datum.timestamp_city_SE3_ego_dict[datum.timestamp_ns])

            pts = datum.sweep.xyz
            city_SE3_ego = datum.timestamp_city_SE3_ego_dict[datum.timestamp_ns]
            points_city = city_SE3_ego.transform_point_cloud(pts)
            points_city = datum.avm.remove_ground_surface(points_city)
            datum.sweep.xyz = city_SE3_ego.inverse().transform_point_cloud(points_city)

        rgba: NDArrayByte = np.full((datum.sweep.xyz.shape[0], 4), dtype=np.uint8, fill_value=32)
        rgba[..., -1] = 255

        if datum.synchronized_imagery is not None:
            for _, v in datum.synchronized_imagery.items():
                uv, _, is_valid = v.camera_model.project_ego_to_img_motion_compensated(
                    datum.sweep.xyz,
                    datum.timestamp_city_SE3_ego_dict[v.timestamp_ns],
                    datum.timestamp_city_SE3_ego_dict[datum.timestamp_ns],
                )
                uv_int: NDArrayInt = np.round(uv).astype(int)[is_valid]  # type: ignore

                rgb_cam = v.img[uv_int[..., 1], uv_int[..., 0]][..., ::-1]
                rgba_cam: NDArrayByte = np.concatenate(  # type: ignore
                    (rgb_cam, np.full_like(rgb_cam[..., -1:], fill_value=255)), axis=-1
                )
                rgba[is_valid] = rgba_cam
        plot += point_cloud(datum.sweep, colors=rgba.tolist())

        plot.show(camera={"pos": [-60, 0, 20], "viewup": [1, 0, 0]}, interactive=False)
        video.addFrame()
        plot.clear()

        if i == 150:
            video.name = f"{datum.log_id}.mp4"
            video.close()
            plot.close()
            break


if __name__ == "__main__":
    render_scene()
