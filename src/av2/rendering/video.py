# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Rendering tools for video visualizations."""

from pathlib import Path
from typing import Dict, Final, Union

import av
import cv2
import numpy as np
import pandas as pd

from av2.utils.typing import NDArrayByte

FFMPEG_OPTIONS: Final[Dict[str, str]] = {"crf": "27"}


def tile_cameras(named_sensors: Dict[str, Union[NDArrayByte, pd.DataFrame]]) -> NDArrayByte:
    """Combine ring cameras into a tiled image.

    NOTE: Images are expected in BGR ordering.

    Layout:

        ##########################################################
        # ring_front_left # ring_front_center # ring_front_right #
        ##########################################################
        # ring_side_left  #                   #  ring_side_right #
        ##########################################################
        ############ ring_rear_left # ring_rear_right ############
        ##########################################################

    Args:
        named_sensors: Dictionary of camera names to the (width, height, 3) images.

    Returns:
        Tiled image.
    """
    landscape_width = 2048
    landscape_height = 1550

    height = landscape_height + landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width
    tiled_im_bgr: NDArrayByte = np.zeros((height, width, 3), dtype=np.uint8)

    ring_rear_left = named_sensors["ring_rear_left"]
    ring_side_left = named_sensors["ring_side_left"]
    ring_front_center = named_sensors["ring_front_center"]
    ring_front_left = named_sensors["ring_front_left"]
    ring_front_right = named_sensors["ring_front_right"]
    ring_side_right = named_sensors["ring_side_right"]
    ring_rear_right = named_sensors["ring_rear_right"]

    tiled_im_bgr[:landscape_height, :landscape_width] = ring_front_left
    tiled_im_bgr[:landscape_width, landscape_width : landscape_width + landscape_height] = ring_front_center
    tiled_im_bgr[:landscape_height, landscape_width + landscape_height :] = ring_front_right

    tiled_im_bgr[landscape_height:3100, :landscape_width] = ring_side_left
    tiled_im_bgr[landscape_height:3100, landscape_width + landscape_height :] = ring_side_right

    start = (width - 4096) // 2
    tiled_im_bgr[3100:4650, start : start + landscape_width] = np.fliplr(ring_rear_left)  # type: ignore
    tiled_im_bgr[3100:4650, start + landscape_width : start + 4096] = np.fliplr(ring_rear_right)  # type: ignore
    tiled_im_rgb: NDArrayByte = cv2.cvtColor(tiled_im_bgr, cv2.COLOR_BGR2RGB)
    return tiled_im_rgb


def write_video(
    video: NDArrayByte,
    dst: Path,
    codec: str = "libx264",
    fps: int = 10,
    crf: int = 27,
    preset: str = "veryfast",
) -> None:
    """Use the FFMPEG Python bindings to encode a video from a sequence of RGB frames.

    Reference: https://github.com/PyAV-Org/PyAV

    Args:
        video: (N,H,W,3) array representing N RGB frames of identical dimensions.
        dst: path to save folder.
        codec: the name of a codec.
        fps: the frame rate for video.
        crf: constant rate factor (CRF) parameter of video, controlling the quality.
            Lower values would result in better quality, at the expense of higher file sizes.
            For x264, the valid Constant Rate Factor (crf) range is 0-51.
        preset: file encoding speed. Options range from "ultrafast", ..., "fast", ..., "medium", ..., "slow", ...
            Higher compression efficiency often translates to slower video encoding speed, at file write time.
    """
    _, H, W, _ = video.shape

    # crop, if the height or width is odd (avoid "height not divisible by 2" error)
    if H % 2 != 0 or W % 2 != 0:
        video = crop_video_to_even_dims(video)
        _, H, W, _ = video.shape

    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(dst), "w") as output:
        stream = output.add_stream(codec, fps)
        stream.width = W
        stream.height = H
        stream.options = {
            "crf": str(crf),
            "hwaccel": "auto",
            "movflags": "+faststart",
            "preset": preset,
            "profile:v": "main",
            "tag": "hvc1",
        }
        for _, img in enumerate(video):
            frame = av.VideoFrame.from_ndarray(img)
            output.mux(stream.encode(frame))
        output.mux(stream.encode(None))


def crop_video_to_even_dims(video: NDArrayByte) -> NDArrayByte:
    """Crop a video tensor (4d nd-array) along the height and width dimensions to assure even dimensions.

    Note: typical "pad" or "crop" filters do not function properly with pypi AV's stream configuration options.

    Args:
        video: (N,H1,W1,3) array representing N RGB frames of identical dimensions, where H1 and W1 may be odd.

    Returns:
        (N,H2,W2,3) array representing N RGB frames of identical dimensions, where H2 and W2 are even.
            The crop is performed on the far right column and/or bottom row of each frame.
    """
    _, H1, W1, _ = video.shape
    height_crop_sz = H1 % 2
    width_crop_sz = W1 % 2

    H2 = H1 - height_crop_sz
    W2 = W1 - width_crop_sz

    cropped_video: NDArrayByte = video[:, :H2, :W2, :]
    return cropped_video
