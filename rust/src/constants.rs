//! # constants
//!
//! Common constants used throughout the library.

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
