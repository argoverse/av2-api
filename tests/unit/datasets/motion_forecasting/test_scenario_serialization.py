# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for Argoverse scenario serialization."""

from pathlib import Path
from typing import List

import numpy as np

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    Track,
    TrackCategory,
)

# Build test ArgoverseScenario
_TEST_OBJECT_STATES: List[ObjectState] = [
    ObjectState(
        observed=True, timestep=0, position=(0.0, 0.0), heading=0.0, velocity=(0.0, 0.0)
    ),
    ObjectState(
        observed=True, timestep=1, position=(1.0, 1.0), heading=0.0, velocity=(1.0, 1.0)
    ),
    ObjectState(
        observed=True, timestep=2, position=(2.0, 2.0), heading=0.0, velocity=(2.0, 2.0)
    ),
]
_TEST_TRACKS: List[Track] = [
    Track(
        track_id="0",
        object_states=_TEST_OBJECT_STATES,
        object_type=ObjectType.VEHICLE,
        category=TrackCategory.FOCAL_TRACK,
    ),
    Track(
        track_id="1",
        object_states=_TEST_OBJECT_STATES,
        object_type=ObjectType.VEHICLE,
        category=TrackCategory.SCORED_TRACK,
    ),
]
_TEST_SCENARIO = ArgoverseScenario(
    scenario_id="test",
    timestamps_ns=np.arange(3),
    tracks=_TEST_TRACKS,
    focal_track_id="test_track",
    city_name="pittsburgh",
    map_id=0,
    slice_id="dummy_slice",
)


def test_parquet_scenario_serialization_roundtrip(tmpdir: Path) -> None:
    """Test serialization and deserialization of Argoverse scenarios in parquet format.

    Args:
        tmpdir: Temp directory used in the test (provided via built-in fixture).
    """
    # Serialize Argoverse scenario to parquet and save to disk
    scenario_path = tmpdir / "test.parquet"
    scenario_serialization.serialize_argoverse_scenario_parquet(
        scenario_path, _TEST_SCENARIO
    )
    assert scenario_path.exists(), "Serialized Argoverse scenario not saved to disk."

    # Check that loading and deserializing a parquet-formatted Argoverse scenario returns an equivalent object
    loaded_test_scenario = scenario_serialization.load_argoverse_scenario_parquet(
        scenario_path
    )
    assert (
        loaded_test_scenario == _TEST_SCENARIO
    ), "Deserialized Argoverse scenario did not match original object."


def test_load_argoverse_scenario_parquet(test_data_root_dir: Path) -> None:
    """Try to load a real scenario from the motion forecasting dataset."""
    test_scenario_id = "0a1e6f0a-1817-4a98-b02e-db8c9327d151"
    test_scenario_path = (
        test_data_root_dir
        / "forecasting_scenarios"
        / test_scenario_id
        / f"scenario_{test_scenario_id}.parquet"
    )

    test_scenario = scenario_serialization.load_argoverse_scenario_parquet(
        test_scenario_path
    )
    assert test_scenario.scenario_id == test_scenario_id
