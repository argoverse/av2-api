# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on rendering tools for video visualizations."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

import av2.rendering.video as video_utils
from av2.utils.typing import NDArrayByte, NDArrayFloat


def generate_dummy_rgb_video(N: int, H: int, W: int) -> NDArrayByte:
    """Generate dummy video data (increasing brightness from top-left to bottom-right, and as video progresses).

    Args:
        N: number of video frames to generate.
        H: frame height, in pixels.
        W: frame width, in pixels.

    Returns:
        tensor of shape (N,H,W,3)

    Raises:
        ValueError: if more than 55 frames are requested (to keep values in [0,200 + 55]).
    """
    if N > 55:
        raise ValueError("Will overflow")

    video: NDArrayByte = np.zeros((N, H, W, 3), dtype=np.uint8)
    for frame_idx in np.arange(N):
        frame_f: NDArrayFloat = np.arange(H * W).reshape(H, W).astype(np.float32)
        frame_f /= frame_f.max()
        frame_f *= 200.0
        frame_f += frame_idx
        frame: NDArrayByte = frame_f.astype(np.uint8)
        for c in range(3):
            video[frame_idx, :, :, c] = frame
    return video


def test_write_video_even_dims() -> None:
    """Ensure we can encode a video tensor (with even H/W dimensions) as a mp4 file with AV, and save it to disk.

    Dummy data is 30 frames of (60,60) RGB video.
    """
    video: NDArrayByte = generate_dummy_rgb_video(N=30, H=60, W=60)
    save_fpath = Path(NamedTemporaryFile(suffix=".mp4").name)
    assert not save_fpath.exists()

    video_utils.write_video(
        video=video,
        dst=save_fpath,
    )
    assert save_fpath.exists()


def test_write_video_odd_dims() -> None:
    """Ensure we can encode a video tensor (with odd H/W dimensions) as a mp4 file with AV, and save it to disk.

    Dummy data is 30 frames of (65,65) RGB video.
    """
    video: NDArrayByte = generate_dummy_rgb_video(N=30, H=65, W=65)
    save_fpath = Path(NamedTemporaryFile(suffix=".mp4").name)
    assert not save_fpath.exists()

    video_utils.write_video(
        video=video,
        dst=save_fpath,
    )
    assert save_fpath.exists()


def test_crop_video_to_even_dims() -> None:
    """Ensure we can crop a video tensor along the height and width dimensions to assure even dimensions.

    Dummy data is 55 frames of (501,501) RGB video.
    """
    video: NDArrayByte = generate_dummy_rgb_video(N=55, H=501, W=501)

    cropped_video = video_utils.crop_video_to_even_dims(video)

    assert cropped_video.shape == (55, 500, 500, 3)
    assert cropped_video.dtype == np.dtype(np.uint8)

    save_fpath = Path(NamedTemporaryFile(suffix=".mp4").name)
    assert not save_fpath.exists()

    video_utils.write_video(
        video=cropped_video, dst=save_fpath, fps=10, preset="medium"
    )
    assert save_fpath.exists()
