//! # constants
//!
//! Common constants used throughout the library.

use strum_macros::{Display, EnumIter, EnumString};

/// Annotation dataframe columns.
/// Found in `annotations.feather`.
pub const ANNOTATION_COLUMNS: [&str; 13] = [
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "num_interior_pts",
    "category",
    "track_uuid",
];

/// Pose dataframe columns.
/// Found in `city_SE3_egovehicle`.
pub const POSE_COLUMNS: [&str; 7] = ["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"];

/// Unknown map file name for use if the map doesn't exist.
pub const DEFAULT_MAP_FILE_NAME: &str = "log_map_archive___DEFAULT_city_00000.json";

/// Argoverse Sensor dataset categories.
#[derive(Clone, Copy, Debug, EnumIter, EnumString, PartialEq)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum AV2Categories {
    /// All recognized animals large enough to affect traffic, but that do not fit into the Cat, Dog, or Horse categories.
    Animal,
    /// Articulated buses perform the same function as a standard city bus, but are able to bend (articulate) towards the center. These will also have a third set of wheels not present on a typical bus.
    ArticulatedBus,
    /// Non-motorized vehicle that typically has two wheels and is propelled by human power pushing pedals in a circular motion.
    Bicycle,
    /// Person actively riding a bicycle, non-pedaling passengers included.
    Bicyclist,
    /// Bollards are short, sturdy posts installed in the roadway or sidewalk to control the flow of traffic. These may be temporary or permanent and are sometimes decorative.
    Bollard,
    /// Chassis cab truck with an enclosed cube shaped cargo area. It should be noted that the cargo area is rigidly attached to the cab, and they do not articulate.
    BoxTruck,
    /// Standard city buses designed to carry a large number of people.
    Bus,
    /// Construction Barrel is a movable traffic barrel that is used to alert drivers to a hazard.  These will typically be orange and white striped and may or may not have a blinking light attached to the top.
    ConstructionBarrel,
    /// Movable traffic cone that is used to alert drivers to a hazard.  These will typically be orange and white striped and may or may not have a blinking light attached to the top.
    ConstructionCone,
    /// Any member of the canine family.
    Dog,
    /// Large motorized vehicles (four wheels or more) which do not fit into any more specific subclass. Examples include extended passenger vans, fire trucks, RVs, etc.
    LargeVehicle,
    /// Trailer carrying a large, mounted, electronic sign to display messages. Often found around construction sites or large events.
    MessageBoardTrailer,
    /// Movable sign designating an area where pedestrians may cross the road.
    MobilePedestrianCrossingSign,
    /// Motorized vehicle with two wheels where the rider straddles the engine.  These are capable of high speeds similar to a car.
    Motorcycle,
    /// Person actively riding a motorcycle or a moped, including passengers.
    Motorcyclist,
    /// Person with authority specifically responsible for stopping and directing vehicles through traffic.
    OfficialSignaler,
    /// Person that is not driving or riding in/on a vehicle. They can be walking, standing, sitting, prone, etc.
    Pedestrian,
    /// Any vehicle that relies on rails to move. This applies to trains, trolleys, train engines, train freight cars, train tanker cars, subways, etc.
    RailedVehicle,
    /// Any conventionally sized passenger vehicle used for the transportation of people and cargo. This includes Cars, vans, pickup trucks, SUVs, etc.
    RegularVehicle,
    /// Bus that primarily holds school children (typically yellow) and can control the flow of traffic via the use of an articulating stop sign and loading/unloading flasher lights.
    SchoolBus,
    /// Official road signs placed by the Department of Transportation (DOT signs) which are of interest to us. This includes yield signs, speed limit signs, directional control signs, construction signs, and other signs that provide required traffic control information. Note that Stop Sign is captured separately and informative signs such as street signs, parking signs, bus stop signs, etc. are not included in this class.
    Sign,
    /// Red octagonal traffic sign displaying the word STOP used to notify drivers that they must come to a complete stop and make sure no other road users are coming before proceeding.
    StopSign,
    /// Push-cart with wheels meant to hold a baby or toddler.
    Stroller,
    /// Mounted, portable traffic light unit commonly used in construction zones or for other temporary detours.
    TrafficLightTrailer,
    /// Vehicles that are clearly defined as a truck but does not fit into the subclasses of Box Truck or Truck Cab. Examples include common delivery vehicles (UPS, FedEx), mail trucks, garbage trucks, utility trucks, ambulances, dump trucks, etc.
    Truck,
    /// Heavy truck commonly known as “Semi cab”, “Tractor”, or “Lorry”. This refers to only the front of part of an articulated tractor trailer.
    TruckCab,
    /// Non-motorized, wheeled vehicle towed behind a motorized vehicle.
    VehicularTrailer,
    /// Chair fitted with wheels for use as a means of transport by a person who is unable to walk as a result of illness, injury, or disability. This includes both motorized and non-motorized wheelchairs as well as low-speed seated scooters not intended for use on the roadway.
    Wheelchair,
    /// Objects involved in the transportation of a person and do not fit a more specific class. Examples range from skateboards, non-motorized scooters, segways, to golf-carts.
    WheeledDevice,
    /// Person actively riding or being carried by a wheeled device.
    WheeledRider,
}

/// Argoverse 2 camera names.
#[derive(Clone, Copy, Debug, Display, EnumIter, EnumString, PartialEq)]
#[strum(serialize_all = "snake_case")]
pub enum CameraNames {
    /// Ring rear left RGB camera.
    RingRearLeft,
    /// Ring side left RGB camera.
    RingSideLeft,
    /// Ring front left RGB camera.
    RingFrontLeft,
    /// Ring front center RGB camera.
    RingFrontCenter,
    /// Ring front right RGB camera.
    RingFrontRight,
    /// Ring side right RGB camera.
    RingSideRight,
    /// Ring rear right RGB camera.
    RingRearRight,
}
