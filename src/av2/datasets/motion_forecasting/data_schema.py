# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Classes that define the data schema for Argoverse motion forecasting scenarios."""

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional, Tuple

from av2.utils.dataclass import dataclass_eq
from av2.utils.typing import NDArrayInt

##########################
# Track-level Data Classes
##########################


@unique
class TrackCategory(Enum):
    """All tracks are categorized with one of four labels, depending on data quality and scenario generation criteria.

    Members:
        TRACK_FRAGMENT: Low quality track that may only contain a few timestamps of observations.
        UNSCORED_TRACK: Track of reasonable quality, but not scored - can be used for contextual input.
        SCORED_TRACK: High-quality tracks relevant to the AV - scored in the multi-agent prediction challenge.
        FOCAL_TRACK: The track used to generate a particular scenario - scored in the single-agent prediction challenge.
    """

    TRACK_FRAGMENT: int = 0
    UNSCORED_TRACK: int = 1
    SCORED_TRACK: int = 2
    FOCAL_TRACK: int = 3


@unique
class ObjectType(str, Enum):
    """All tracks are assigned one of the following object type labels."""

    # Dynamic movers
    VEHICLE: str = "vehicle"
    PEDESTRIAN: str = "pedestrian"
    MOTORCYCLIST: str = "motorcyclist"
    CYCLIST: str = "cyclist"
    BUS: str = "bus"

    # Static objects
    STATIC: str = "static"
    BACKGROUND: str = "background"
    CONSTRUCTION: str = "construction"
    RIDERLESS_BICYCLE: str = "riderless_bicycle"

    # Catch-all type for other/unknown objects
    UNKNOWN: str = "unknown"


@dataclass
class ObjectState:
    """Bundles all state information associated with an object at a fixed point in time.

    Attributes:
        observed: Boolean indicating if this object state falls in the observed segment of the scenario.
        timestep: Time step corresponding to this object state [0, num_scenario_timesteps).
        position: (x, y) Coordinates of center of object bounding box.
        heading: Heading associated with object bounding box (in radians, defined w.r.t the map coordinate frame).
        velocity: (x, y) Instantaneous velocity associated with the object (in m/s).
    """

    observed: bool
    timestep: int
    position: Tuple[float, float]
    heading: float
    velocity: Tuple[float, float]


@dataclass(eq=False)
class Track:
    """Bundles all data associated with an Argoverse track.

    Attributes:
        track_id: Unique ID associated with this track.
        object_states: States for each timestep where the track object had a valid observation.
        object_type: Inferred type for the track object.
        category: Assigned category for track - used as an indicator for prediction requirements and data quality.
    """

    track_id: str
    object_states: List[ObjectState]
    object_type: ObjectType
    category: TrackCategory

    def __eq__(self, other: object) -> bool:
        """Dataclass equality operator that correctly handles data members."""
        return dataclass_eq(self, other)


#############################
# Scenario-level data classes
#############################


@dataclass(eq=False)
class ArgoverseScenario:
    """Bundles all data associated with an Argoverse scenario.

    Attributes:
        scenario_id: Unique ID associated with this scenario.
        timestamps_ns: All timestamps associated with this scenario.
        tracks: All tracks associated with this scenario.
        focal_track_id: The track ID associated with the focal agent of the scenario.
        city_name: The name of the city associated with this scenario.
        map_id: The map ID associated with the scenario (used for internal bookkeeping).
        slice_id: ID of the slice used to generate the scenario (used for internal bookkeeping).
    """

    scenario_id: str
    timestamps_ns: NDArrayInt
    tracks: List[Track]
    focal_track_id: str
    city_name: str
    map_id: Optional[int]
    slice_id: Optional[str]

    def __eq__(self, other: object) -> bool:
        """Dataclass equality operator that correctly handles data members."""
        return dataclass_eq(self, other)
