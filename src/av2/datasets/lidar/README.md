# Argoverse 2 Lidar Dataset Overview

The Argoverse 2 Lidar Dataset is intended to support research into self-supervised learning in the lidar domain as well as point cloud forecasting. The AV2 Lidar Dataset is mined with the same criteria as the Forecasting Dataset to ensure that each scene is interesting. While the Lidar Dataset does not have 3D object annotations, each scenario carries an HD map with rich, 3D information about the scene.

## Dataset Size

Our dataset is the largest such collection to date with 20,000 thirty second sequences.

## Sensor Suite

Lidar sweeps are collected at 10 Hz. In addition, 6-DOF ego-vehicle pose in a global coordinate system are provided. Lidar returns are captured by two 32-beam lidars, spinning at 10 Hz in the same direction, but separated in orientation by 180°.

## Dataset Structure Format

Tabular data (lidar sweeps, poses) are provided as [Apache Feather Files](https://arrow.apache.org/docs/python/feather.html) with the file extension `.feather`.

A local map is provided per log.

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

An example sweep sensors/lidar/15970913559644000.feather, meaning a reference timestamp of 15970913559644000 nanoseconds:
```
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
