"""Scene Flow evaluation unit tests."""

import tempfile

from av2.evaluation.scene_flow.example_submission import example_submission
from av2.evaluation.scene_flow.make_annotation_files import make_annotation_files
from av2.evaluation.scene_flow.make_mask_files import make_mask_files

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent


def test_submission():
    # Pipeline:
    # 1. make mask files and produce masks.zip
    # 2. make annotation files using masks.zip
    # 3. make format file? maybe just eliminate this and use masks.zip
    # 3. make example submission files
    # 3. make submission archive from submission files

    # TODO
    # - add zipping step to make_mask_files
    # - DONE change get_eval_point_mask so that it can handle a temporary mask file
    # - DONE change make_submission_archive to use just masks.zip and allow overriding the default mask file
    dl_test = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "test")
    data_loader = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "val")

    with Path(tempfile.TemporaryDirectory().name) as test_dir:
        mask_dir = test_dir / "masks"
