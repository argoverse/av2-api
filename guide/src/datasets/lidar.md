# Argoverse 2 Lidar Dataset Overview

<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158715494-472339d1-a5d5-4d33-8fcf-3455c0d78d27.gif" height="180">
  <img src="https://user-images.githubusercontent.com/29715011/158715496-f439ccad-71af-4880-8b43-ade7b6c8f333.gif" height="180">
  <img src="https://user-images.githubusercontent.com/29715011/158715498-23d7a11f-12a1-4aeb-b9af-dbced217b340.gif" height="180">
  <img src="https://user-images.githubusercontent.com/29715011/158715497-d1603423-c32f-4cf0-ab1e-6bbc9c458535.gif" height="180">
</p>

## Table of Contents

<!-- toc -->

## Overview

The Argoverse 2 Lidar Dataset is intended to support research into self-supervised learning in the lidar domain as well as point cloud forecasting. The AV2 Lidar Dataset is mined with the same criteria as the Forecasting Dataset to ensure that each scene is interesting. While the Lidar Dataset does not have 3D object annotations, each scenario carries an HD map with rich, 3D information about the scene.

## Dataset Size

Our dataset is the largest such collection to date with 20,000 thirty second sequences.

## Sensor Suite

Lidar sweeps are collected at 10 Hz. In addition, 6-DOF ego-vehicle pose in a global coordinate system are provided. Lidar returns are captured by two 32-beam lidars, spinning at 10 Hz in the same direction, but separated in orientation by 180°.

We aggregate all returns from the two stacked 32-beam sensors into a single sweep. These sensors each have different, overlapping fields-of-view. Both lidars have their own reference frame, and we refer to them as `up_lidar` and `down_lidar`, respectively. We have egomotion-compensated the lidar sensor data to the egovehicle reference nanosecond timestamp. **All lidar returns are provided in the egovehicle reference frame, not the individual lidar reference frame**.

## Dataset Structure Format

Tabular data (lidar sweeps, poses) are provided as [Apache Feather Files](https://arrow.apache.org/docs/python/feather.html) with the file extension `.feather`.

**Maps:** A local vector map is provided per log, please refer to the [Map README](../../map/README.md) for additional details.

Directory structure:
```
av2
└───lidar
    └───train
    |   └───LyIXwbWeHWPHYUZjD1JPdXcvvtYumCWG
    |       └───sensors
    |       |   └───lidar
    |       |       └───15970913559644000.feather
    |       |                      .
    |       |                      .
    |       |                      .
    |       └───calibration
    |       |   └───egovehicle_SE3_sensor.feather
    |       └───map
    |       |   └───log_map_archive_LyIXwbWeHWPHYUZjD1JPdXcvvtYumCWG__Summer____PIT_city_77257.json
    |       └───city_SE3_egovehicle.feather
    └───val
    └───test
```

An example sweep `sensors/lidar/15970913559644000.feather`, meaning a reference timestamp of 15970913559644000 nanoseconds:
```python
               x          y         z  intensity  laser_number  offset_ns
0      -1.291016   2.992188 -0.229370         24            31    3318000
1     -25.921875  25.171875  0.992188          5            14    3318000
2     -15.500000  18.937500  0.901855         34            16    3320303
3      -3.140625   4.593750 -0.163696         12            30    3320303
4      -4.445312   6.535156 -0.109802         14            29    3322607
...          ...        ...       ...        ...           ...        ...
98231  18.312500 -38.187500  3.279297         26            50  106985185
98232  23.109375 -34.437500  3.003906         20            49  106987490
98233   4.941406  -5.777344 -0.162720         12            32  106987490
98234   6.640625  -8.257812 -0.157593          6            33  106989794
98235  20.015625 -37.062500  2.550781         12            47  106989794

[98236 rows x 6 columns]
```

## Lidar Dataset splits
We randomly partition the dataset into the following splits:

- Train (16,000 logs)
- Validation (2,000 logs)
- Test (2,000 logs)