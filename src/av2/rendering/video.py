# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Rendering tools for video visualizations."""

from __future__ import annotations

from enum import Enum, unique
from pathlib import Path
from typing import Dict, Final, Mapping, Optional, Set, Union

import av
import numpy as np
import pandas as pd

from av2.rendering.color import ColorFormats
from av2.utils.typing import NDArrayByte

COLOR_FORMAT_TO_PYAV_COLOR_FORMAT: Final[Dict[ColorFormats, str]] = {
    ColorFormats.RGB: "rgb24",
    ColorFormats.BGR: "bgr24",
}
FFMPEG_OPTIONS: Final[Dict[str, str]] = {"crf": "27"}


@unique
class VideoCodecs(str, Enum):
    """Available video codecs for encoding mp4 videos.

    NOTE: The codecs available are dependent on the FFmpeg build that
        you are using. We recommend defaulting to LIBX264.
    """

    LIBX264 = "libx264"  # https://en.wikipedia.org/wiki/Advanced_Video_Coding
    LIBX265 = "libx265"  # https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding
    HEVC_VIDEOTOOLBOX = "hevc_videotoolbox"  # macOS GPU acceleration.


HIGH_EFFICIENCY_VIDEO_CODECS: Final[Set[VideoCodecs]] = set(
    [VideoCodecs.LIBX265, VideoCodecs.HEVC_VIDEOTOOLBOX]
)


def tile_cameras(
    named_sensors: Mapping[str, Union[NDArrayByte, pd.DataFrame]],
    bev_img: Optional[NDArrayByte] = None,
) -> NDArrayByte:
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
        bev_img: (H,W,3) Bird's-eye view image.

    Returns:
        Tiled image.
    """
    landscape_height = 2048
    landscape_width = 1550
    for _, v in named_sensors.items():
        landscape_width = max(v.shape[0], v.shape[1])
        landscape_height = min(v.shape[0], v.shape[1])
        break

    height = landscape_height + landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width
    tiled_im_bgr: NDArrayByte = np.zeros((height, width, 3), dtype=np.uint8)

    if "ring_front_left" in named_sensors:
        ring_front_left = named_sensors["ring_front_left"]
        tiled_im_bgr[:landscape_height, :landscape_width] = ring_front_left

    if "ring_front_center" in named_sensors:
        ring_front_center = named_sensors["ring_front_center"]
        tiled_im_bgr[
            :landscape_width, landscape_width : landscape_width + landscape_height
        ] = ring_front_center

    if "ring_front_right" in named_sensors:
        ring_front_right = named_sensors["ring_front_right"]
        tiled_im_bgr[
            :landscape_height, landscape_width + landscape_height :
        ] = ring_front_right

    if "ring_side_left" in named_sensors:
        ring_side_left = named_sensors["ring_side_left"]
        tiled_im_bgr[
            landscape_height : 2 * landscape_height, :landscape_width
        ] = ring_side_left

    if "ring_side_right" in named_sensors:
        ring_side_right = named_sensors["ring_side_right"]
        tiled_im_bgr[
            landscape_height : 2 * landscape_height,
            landscape_width + landscape_height :,
        ] = ring_side_right

    if bev_img is not None:
        tiled_im_bgr[
            landscape_width : 2 * landscape_width,
            landscape_width : landscape_width + landscape_height,
        ] = bev_img

    if "ring_rear_left" in named_sensors:
        ring_rear_left = named_sensors["ring_rear_left"]
        tiled_im_bgr[
            2 * landscape_height : 3 * landscape_height, :landscape_width
        ] = ring_rear_left

    if "ring_rear_right" in named_sensors:
        ring_rear_right = named_sensors["ring_rear_right"]
        tiled_im_bgr[
            2 * landscape_height : 3 * landscape_height, width - landscape_width :
        ] = ring_rear_right
    return tiled_im_bgr


def write_video(
    video: NDArrayByte,
    dst: Path,
    color_format: ColorFormats = ColorFormats.RGB,
    codec: VideoCodecs = VideoCodecs.LIBX264,
    fps: int = 10,
    crf: int = 27,
    preset: str = "veryfast",
) -> None:
    """Use the FFMPEG Python bindings to encode a video from a sequence of RGB frames.

    Reference: https://github.com/PyAV-Org/PyAV

    Args:
        video: (N,H,W,3) Array representing N RGB frames of identical dimensions.
        dst: Path to save folder.
        color_format: Format of the color channels.
        codec: Name of the codec.
        fps: Frame rate for video.
        crf: Constant rate factor (CRF) parameter of video, controlling the quality.
            Lower values would result in better quality, at the expense of higher file sizes.
            For x264, the valid Constant Rate Factor (crf) range is 0-51.
        preset: File encoding speed. Options range from "ultrafast", ..., "fast", ..., "medium", ..., "slow", ...
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
        if codec in HIGH_EFFICIENCY_VIDEO_CODECS:
            stream.codec_tag = "hvc1"
        stream.width = W
        stream.height = H
        stream.options = {
            "crf": str(crf),
            "hwaccel": "auto",
            "movflags": "+faststart",
            "preset": preset,
            "profile:v": "main",
        }

        format = COLOR_FORMAT_TO_PYAV_COLOR_FORMAT[color_format]
        for _, img in enumerate(video):
            frame = av.VideoFrame.from_ndarray(img, format=format)
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
