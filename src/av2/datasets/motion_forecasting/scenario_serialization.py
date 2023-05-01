# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utilities to (de)serialize Argoverse forecasting scenarios."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    Track,
    TrackCategory,
)


def serialize_argoverse_scenario_parquet(
    save_path: Path, scenario: ArgoverseScenario
) -> None:
    """Serialize a single Argoverse scenario in parquet format and save to disk.

    Args:
        save_path: Path where the scenario should be saved.
        scenario: Scenario to serialize and save.
    """
    # Convert scenario data to DataFrame
    tracks_df = _convert_tracks_to_tabular_format(scenario.tracks)
    tracks_df["scenario_id"] = scenario.scenario_id
    tracks_df["start_timestamp"] = scenario.timestamps_ns[0]
    tracks_df["end_timestamp"] = scenario.timestamps_ns[-1]
    tracks_df["num_timestamps"] = len(scenario.timestamps_ns)
    tracks_df["focal_track_id"] = scenario.focal_track_id
    tracks_df["city"] = scenario.city_name

    if scenario.map_id is not None:
        tracks_df["map_id"] = scenario.map_id
    if scenario.slice_id is not None:
        tracks_df["slice_id"] = scenario.slice_id

    # Serialize the scenario dataframe to a parquet file
    tracks_df.to_parquet(save_path)


def load_argoverse_scenario_parquet(scenario_path: Path) -> ArgoverseScenario:
    """Load a serialized Argoverse scenario from disk.

    Args:
        scenario_path: Path to the saved scenario in parquet format.

    Raises:
        FileNotFoundError: If no file exists at the specified scenario_path.

    Returns:
        Preprocessed scenario object that was saved at scenario_path.
    """
    if not Path(scenario_path).exists():
        raise FileNotFoundError(f"No scenario exists at location: {scenario_path}.")

    # Load scenario dataframe from parquet file
    tracks_df = pd.read_parquet(scenario_path)

    # Read track data and scenario metadata from scenario dataframe
    tracks = _load_tracks_from_tabular_format(tracks_df)
    scenario_id = tracks_df["scenario_id"][0]
    focal_track_id = tracks_df["focal_track_id"][0]
    city_name = tracks_df["city"][0]
    map_id = tracks_df["map_id"][0] if "map_id" in tracks_df.columns else None
    slice_id = tracks_df["slice_id"][0] if "slice_id" in tracks_df.columns else None

    # Interpolate scenario timestamps based on the saved start and end timestamps
    timestamps_ns = np.linspace(
        tracks_df["start_timestamp"][0],
        tracks_df["end_timestamp"][0],
        num=tracks_df["num_timestamps"][0],
    )

    return ArgoverseScenario(
        scenario_id=scenario_id,
        timestamps_ns=timestamps_ns,
        tracks=tracks,
        focal_track_id=focal_track_id,
        city_name=city_name,
        map_id=map_id,
        slice_id=slice_id,
    )


def _convert_tracks_to_tabular_format(tracks: List[Track]) -> pd.DataFrame:
    """Convert tracks to tabular data format.

    Args:
        tracks: All tracks associated with the scenario.

    Returns:
        DataFrame containing all track data in a tabular format.
    """
    track_dfs: List[pd.DataFrame] = []

    for track in tracks:
        track_df = pd.DataFrame()

        observed_states: List[bool] = []
        timesteps: List[int] = []
        positions_x: List[float] = []
        positions_y: List[float] = []
        headings: List[float] = []
        velocities_x: List[float] = []
        velocities_y: List[float] = []

        for object_state in track.object_states:
            observed_states.append(object_state.observed)
            timesteps.append(object_state.timestep)
            positions_x.append(object_state.position[0])
            positions_y.append(object_state.position[1])
            headings.append(object_state.heading)
            velocities_x.append(object_state.velocity[0])
            velocities_y.append(object_state.velocity[1])

        track_df["observed"] = observed_states
        track_df["track_id"] = track.track_id
        track_df["object_type"] = track.object_type.value
        track_df["object_category"] = track.category.value
        track_df["timestep"] = timesteps
        track_df["position_x"] = positions_x
        track_df["position_y"] = positions_y
        track_df["heading"] = headings
        track_df["velocity_x"] = velocities_x
        track_df["velocity_y"] = velocities_y

        track_dfs.append(track_df)

    return pd.concat(track_dfs, ignore_index=True)


def _load_tracks_from_tabular_format(tracks_df: pd.DataFrame) -> List[Track]:
    """Load tracks from tabular data format.

    Args:
        tracks_df: DataFrame containing all track data in a tabular format.

    Returns:
        All tracks associated with the scenario.
    """
    tracks: List[Track] = []

    for track_id, track_df in tracks_df.groupby("track_id"):
        observed_states: List[bool] = track_df.loc[:, "observed"].values.tolist()
        object_type: ObjectType = ObjectType(track_df["object_type"].iloc[0])
        object_category: TrackCategory = TrackCategory(
            track_df["object_category"].iloc[0]
        )
        timesteps: List[int] = track_df.loc[:, "timestep"].values.tolist()
        positions: List[Tuple[float, float]] = list(
            zip(
                track_df.loc[:, "position_x"].values.tolist(),
                track_df.loc[:, "position_y"].values.tolist(),
            )
        )
        headings: List[float] = track_df.loc[:, "heading"].values.tolist()
        velocities: List[Tuple[float, float]] = list(
            zip(
                track_df.loc[:, "velocity_x"].values.tolist(),
                track_df.loc[:, "velocity_y"].values.tolist(),
            )
        )

        object_states: List[ObjectState] = []
        for idx in range(len(timesteps)):
            object_states.append(
                ObjectState(
                    observed=observed_states[idx],
                    timestep=timesteps[idx],
                    position=positions[idx],
                    heading=headings[idx],
                    velocity=velocities[idx],
                )
            )

        tracks.append(
            Track(
                track_id=track_id,
                object_states=object_states,
                object_type=object_type,
                category=object_category,
            )
        )

    return tracks
