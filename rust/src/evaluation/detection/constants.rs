/// Constants used in the detection evaluation.
use strum_macros::EnumString;

/// Unique identifier columns.
pub const UUID_COLUMNS: [&str; 3] = ["log_id", "timestamp_ns", "category"];
/// Max scale error.
pub const MAX_SCALE_ERROR: f32 = 1.0;
/// Max yaw error in radians.
pub const MAX_YAW_RAD_ERROR: f32 = std::f32::consts::PI;

/// Minimum average precision.
pub const MIN_AP: f32 = 0.0;
/// Minimum composite detection score.
pub const MIN_CDS: f32 = 0.0;

/// Maximum normalized error average translation error.
pub const MAX_NORMALIZED_ATE: f32 = 1.0;
/// Maximum normalized error average scaling error.
pub const MAX_NORMALIZED_ASE: f32 = 1.0;
/// Maximum normalized error average orientation error.
pub const MAX_NORMALIZED_AOE: f32 = 1.0;

/// Ordered columns for cuboid detections.
pub const ORDERED_CUBOID_COLUMNS: [&str; 11] = [
    "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz", "score",
];

/// Detection competition categories.
#[derive(Clone, Copy, Debug, PartialEq, EnumString)]
pub enum DetectionCompetitionCategories {
    ArticulatedBus,
    Bicycle,
    Bicyclist,
    Bollard,
    BoxTruck,
    Bus,
    ConstructionBarrel,
    ConstructionCone,
    Dog,
    LargeVehicle,
    MessageBoardTrailer,
    MobilePedestrianCrossingSign,
    Motorcycle,
    Motorcyclist,
    Pedestrian,
    RegularVehicle,
    SchoolBus,
    Sign,
    StopSign,
    Stroller,
    Truck,
    TruckCab,
    VehicularTrailer,
    WheelChair,
    WheeledDevice,
    WheeledRider,
}

/// Affinity types.
#[derive(Clone, Copy, Debug, PartialEq, EnumString)]
pub enum AffinityType {
    /// 3D Euclidean distance-based affinity between cuboid centers.
    CENTER,
}

/// Filter metric types.
#[derive(Clone, Copy, Debug, PartialEq, EnumString)]
pub enum FilterMetricType {
    /// 3D Euclidean distance-based filtering.
    EUCLIDEAN,
}
