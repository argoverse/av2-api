"""Unit tests on scene_flow data loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import av2.torch.data_loaders.scene_flow
from av2.evaluation.scene_flow.utils import get_eval_point_mask, get_eval_subset
from av2.torch.structures.flow import Flow
from av2.torch.structures.sweep import Sweep

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent


def test_scene_flow_dataloader() -> None:
    """Test loading a single pair with flow annotations.

    The computed flow should check the visually confirmed labels in flow_labels.feather.
    """
    dl_test = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "test")
    sweep_0, sweep_1, ego, not_flow = dl_test[0]
    assert not_flow is None
    rust_sweep = dl_test._backend.get(0)
    sweep_0 = Sweep.from_rust(rust_sweep)
    assert sweep_0.cuboids is None

    failed = False
    try:
        get_eval_point_mask((sweep_0, sweep_1, ego, not_flow))
    except ValueError:
        failed = True
    assert failed

    failed = False
    try:
        Flow.from_sweep_pair((sweep_0, sweep_1))
    except ValueError:
        failed = True
    assert failed

    dl = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(_TEST_DATA_ROOT, "test_data", "val")
    assert len(dl) == 1
    assert dl.get_log_id(0) == "7fab2350-7eaf-3b7e-a39d-6937a4c1bede"

    for datum in dl:
        sweep_0, sweep_1, ego, maybe_flow = datum

    assert maybe_flow is not None
    flow: Flow = maybe_flow
    assert len(flow) == len(sweep_0.lidar.as_tensor())

    log_dir = _TEST_DATA_ROOT / "test_data/sensor/val/7fab2350-7eaf-3b7e-a39d-6937a4c1bede"
    flow_labels = pd.read_feather(log_dir / "flow_labels.feather")

    FLOW_COLS = ["flow_tx_m", "flow_ty_m", "flow_tz_m"]
    assert np.allclose(flow.flow.numpy(), flow_labels[FLOW_COLS].to_numpy(), atol=1e-3)
    assert np.allclose(flow.classes.numpy(), flow_labels.classes.to_numpy())
    assert np.allclose(flow.dynamic.numpy(), flow_labels.dynamic.to_numpy())
    assert sweep_0.is_ground is not None
    ground_match: NDArrayBool = sweep_0.is_ground.numpy() == flow_labels.is_ground_0.to_numpy()
    assert np.all(ground_match), f"{ground_match.sum()} differences: {list(ground_match)}"

    gt_ego = np.load(log_dir / "ego_motion.npz")

    assert np.allclose(ego.matrix().numpy(), gt_ego["ego_motion"], atol=1e-3)

    eval_inds = get_eval_subset(dl)
    assert len(eval_inds) == 1
    assert eval_inds[0] == 0

    eval_mask = get_eval_point_mask((sweep_0, sweep_1, ego, flow))

    pcl = sweep_0.lidar.as_tensor()[:, :3]
    is_close = torch.logical_and(pcl[:, 0].abs() <= 50, pcl[:, 1].abs() <= 50).bool()
    not_ground = torch.logical_not(sweep_0.is_ground)
    assert (eval_mask == (torch.logical_and(is_close, not_ground))).all()
