"""Unit tests on scene_flow data-loader."""

from pathlib import Path

import numpy as np
import pandas as pd

import av2.torch.data_loaders.scene_flow
from av2.evaluation.scene_flow.utils import compute_eval_point_mask, get_eval_subset
from av2.torch.structures.flow import Flow
from av2.torch.structures.sweep import Sweep
from av2.utils.typing import NDArrayBool

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent


def test_scene_flow_dataloader() -> None:
    """Test loading a single pair with flow annotations.

    The computed flow should check the visually confirmed labels in flow_labels.feather.
    """
    dl_test = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(
        _TEST_DATA_ROOT, "test_data", "test"
    )
    sweep_0, sweep_1, ego, not_flow = dl_test[0]
    assert not_flow is None
    rust_sweep = dl_test._backend.get(0)
    sweep_0 = Sweep.from_rust(rust_sweep)
    assert sweep_0.cuboids is None

    failed = False
    try:
        compute_eval_point_mask((sweep_0, sweep_1, ego, not_flow))
    except ValueError:
        failed = True
    assert failed

    failed = False
    try:
        Flow.from_sweep_pair((sweep_0, sweep_1))
    except ValueError:
        failed = True
    assert failed

    data_loader = av2.torch.data_loaders.scene_flow.SceneFlowDataloader(
        _TEST_DATA_ROOT, "test_data", "val"
    )
    assert len(data_loader) == 1
    assert data_loader.get_log_id(0) == "7fab2350-7eaf-3b7e-a39d-6937a4c1bede"

    for datum in data_loader:
        sweep_0, sweep_1, ego, maybe_flow = datum

    assert maybe_flow is not None
    flow: Flow = maybe_flow
    assert len(flow) == len(sweep_0.lidar.as_tensor())

    log_dir = (
        _TEST_DATA_ROOT / "test_data/sensor/val/7fab2350-7eaf-3b7e-a39d-6937a4c1bede"
    )
    flow_labels = pd.read_feather(log_dir / "flow_labels.feather")

    FLOW_COLS = ["flow_tx_m", "flow_ty_m", "flow_tz_m"]
    err = np.abs(flow.flow.numpy() - flow_labels[FLOW_COLS].to_numpy()).sum(-1)
    max_err_ind = np.argmax(err)
    flow_err_val = flow.flow.numpy()[max_err_ind]
    label_err_val = flow_labels[FLOW_COLS].to_numpy()[max_err_ind]
    assert np.allclose(
        flow.flow.numpy(), flow_labels[FLOW_COLS].to_numpy(), atol=1e-3
    ), f"max-diff {err[max_err_ind]} ind: {max_err_ind} flow: {flow_err_val} label: {label_err_val}"
    assert np.allclose(flow.category_indices.numpy(), flow_labels.classes.to_numpy())
    assert np.allclose(flow.is_dynamic.numpy(), flow_labels.dynamic.to_numpy())
    assert sweep_0.is_ground is not None
    ground_match: NDArrayBool = (
        sweep_0.is_ground.numpy() == flow_labels.is_ground_0.to_numpy()
    )
    assert np.logical_not(ground_match).sum() < 10

    gt_ego = np.load(log_dir / "ego_motion.npz")

    assert np.allclose(ego.matrix().numpy(), gt_ego["ego_motion"], atol=1e-3)

    eval_inds = get_eval_subset(data_loader)
    assert len(eval_inds) == 1
    assert eval_inds[0] == 0
