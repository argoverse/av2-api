# Overview

<div align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158486284-1a0df794-ee0a-4ae6-a320-0dd0d1daad06.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/158486286-e734e654-b879-4994-a129-9957cc591af4.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/158486288-5e7c0971-de0c-4ff5-bea7-76f7922dd1e0.gif" height="225">
</div>

## Table of Contents

<!-- toc -->

## Overview

The Argoverse 2 motion forecasting dataset consists of 250,000 scenarios, collected from 6 cities spanning multiple seasons.

Each scenario is specifically designed to maximize interactions relevant to the ego-vehicle. This naturally results in the inclusion of actor-dense scenes featuring a range of vehicle and non-vehicle actor types. At the time of release, AV2 provides the largest object taxonomy, in addition to the broadest mapped area of any motion forecasting dataset released so far.

## Download

The latest version of the AV2 motion forecasting dataset can be downloaded from the Argoverse [website](https://www.argoverse.org/av2.html).

## Scenarios and Tracks

Each scenario is 11s long and consists of a collection of actor histories, which are represented as "tracks". For each scenario, we provide the following high-level attributes:

- `scenario_id`: Unique ID associated with this scenario.
- `timestamps_ns`: All timestamps associated with this scenario.
- `tracks`: All tracks associated with this scenario.
- `focal_track_id`: The track ID associated with the focal agent of the scenario.
- `city_name`: The name of the city associated with this scenario.

Each track is further associated with the following attributes:

- `track_id`: Unique ID associated with this track
- `object_states`: States for each timestep where the track object had a valid observation.
- `object_type`: Inferred type for the track object.
- `category`: Assigned category for track - used as an indicator for prediction requirements and data quality.

Track object states bundle all information associated with a particular actor at a fixed point in time:

- `observed`: Boolean indicating if this object state falls in the observed segment of the scenario.
- `timestep`: Time step corresponding to this object state [0, num_scenario_timesteps).
- `position`: (x, y) Coordinates of center of object bounding box.
- `heading`: Heading associated with object bounding box (in radians, defined w.r.t the map coordinate frame).
- `velocity`: (x, y) Instantaneous velocity associated with the object (in m/s).

Each track is assigned one of the following labels, which dictate scoring behavior in the Argoverse challenges:

- `TRACK_FRAGMENT`: Lower quality track that may only contain a few timestamps of observations.
- `UNSCORED_TRACK`: Unscored track used for contextual input.
- `SCORED_TRACK`: High-quality tracks relevant to the AV - scored in the multi-agent prediction challenge.
- `FOCAL_TRACK`: The primary track of interest in a given scenario - scored in the single-agent prediction challenge.

Each track is also assigned one of the following labels, as part of the 10-class object taxonomy:

- Dynamic
  - `VEHICLE`
  - `PEDESTRIAN`
  - `MOTORCYCLIST`
  - `CYCLIST`
  - `BUS`
- Static
  - `STATIC`
  - `BACKGROUND`
  - `CONSTRUCTION`
  - `RIDERLESS_BICYCLE`
- `UNKNOWN`

For more additional details regarding the data schema, please refer [here](data_schema.py).

## Visualization

Motion forecasting scenarios can be visualized using the viz [`script`](../../../../tutorials/generate_forecasting_scenario_visualizations.py) or by calling the viz [`library`](viz/scenario_visualization.py#L48) directly.
