# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Constants relevant to the sensors and log data of the Argoverse 2 Sensor Dataset."""

from enum import Enum, unique


@unique
class RingCameras(str, Enum):
    """Ring Camera names."""

    RING_REAR_LEFT = "ring_rear_left"
    RING_SIDE_LEFT = "ring_side_left"
    RING_FRONT_LEFT = "ring_front_left"
    RING_FRONT_CENTER = "ring_front_center"
    RING_FRONT_RIGHT = "ring_front_right"
    RING_SIDE_RIGHT = "ring_side_right"
    RING_REAR_RIGHT = "ring_rear_right"


@unique
class StereoCameras(str, Enum):
    """Stereo camera names."""

    STEREO_FRONT_LEFT = "stereo_front_left"
    STEREO_FRONT_RIGHT = "stereo_front_right"


@unique
class Cameras(str, Enum):
    """All cameras."""

    RING_CAMERAS = RingCameras
    STEREO_CAMERAS = StereoCameras


@unique
class AnnotationCategories(str, Enum):
    """Sensor dataset annotation categories."""

    ANIMAL = "ANIMAL"
    ARTICULATED_BUS = "ARTICULATED_BUS"
    BICYCLE = "BICYCLE"
    BICYCLIST = "BICYCLIST"
    BOLLARD = "BOLLARD"
    BOX_TRUCK = "BOX_TRUCK"
    BUS = "BUS"
    CONSTRUCTION_BARREL = "CONSTRUCTION_BARREL"
    CONSTRUCTION_CONE = "CONSTRUCTION_CONE"
    DOG = "DOG"
    LARGE_VEHICLE = "LARGE_VEHICLE"
    MESSAGE_BOARD_TRAILER = "MESSAGE_BOARD_TRAILER"
    MOBILE_PEDESTRIAN_CROSSING_SIGN = "MOBILE_PEDESTRIAN_CROSSING_SIGN"
    MOTORCYCLE = "MOTORCYCLE"
    MOTORCYCLIST = "MOTORCYCLIST"
    OFFICIAL_SIGNALER = "OFFICIAL_SIGNALER"
    PEDESTRIAN = "PEDESTRIAN"
    RAILED_VEHICLE = "RAILED_VEHICLE"
    REGULAR_VEHICLE = "REGULAR_VEHICLE"
    SCHOOL_BUS = "SCHOOL_BUS"
    SIGN = "SIGN"
    STOP_SIGN = "STOP_SIGN"
    STROLLER = "STROLLER"
    TRAFFIC_LIGHT_TRAILER = "TRAFFIC_LIGHT_TRAILER"
    TRUCK = "TRUCK"
    TRUCK_CAB = "TRUCK_CAB"
    VEHICULAR_TRAILER = "VEHICULAR_TRAILER"
    WHEELCHAIR = "WHEELCHAIR"
    WHEELED_DEVICE = "WHEELED_DEVICE"
    WHEELED_RIDER = "WHEELED_RIDER"
