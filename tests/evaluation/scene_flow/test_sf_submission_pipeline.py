"""Scene Flow evaluation unit tests."""

import tempfile

from av2.evaluation.scene_flow.example_submission import example_submission
from av2.evaluation.scene_flow.make_annotation_files import make_annotation_files
from av2.evaluation.scene_flow.make_mask_files import make_mask_files

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent


def test_submission():
    dl_test = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "test")
    data_loader = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "val")

    with Path(tempfile.TemporaryDirectory().name) as test_dir:
        mask_dir = test_dir / "masks"
