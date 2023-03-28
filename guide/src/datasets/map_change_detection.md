# Overview

<div align="center">
  <img src="https://user-images.githubusercontent.com/29715011/159289930-a58147c3-c6ed-4b4e-a2a8-e23c23feb43e.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/159289891-8aae12e7-136a-4f44-bbc1-8ef93f01e23e.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/159152108-3c3001fe-ec7c-48fd-8c08-4a473affb2a3.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/159152102-27c04180-9ca4-4725-be81-95ee6858d367.gif" height="225">
</div>

## Table of Contents

<!-- toc -->

## Overview

The Trust, but Verify (TbV) Dataset consists of 1043 vehicle logs. Each vehicle log, on average, is 54 seconds in duration, including 536 LiDAR sweeps on average, and 1073 images from each of the 7 cameras (7512 images per log). Some logs are as short as 4 seconds, and other logs are up to 117 seconds in duration.

The total dataset amounts to 15.54 hours of driving data, amounting to 922 GB of data in its extracted form. There are 7.84 Million images in the dataset (7,837,614 exactly), and 559,440 LiDAR sweeps in total.


## Downloading TbV

TbV is available for download in two forms -- either zipped up as 21 tar.gz files -- or in extracted, unzipped form (without tar archives). Downloading either will produce the same result (the underlying log data is identical).

Using the `tar.gz` files is recommended (depending upon your connection, this is likely faster, as there are almost 8 million images files in the extracted format). We recommend using `s5cmd` to pull down all 21 `.tar.gz` files with a single command. You can see the links to the `tar.gz` files on [the Argoverse 2 downloads page](https://www.argoverse.org/av2.html#download-link).

First, install `s5cmd` using [the installation instructions here](https://github.com/argoai/argoverse2-api/blob/main/DOWNLOAD.md), and then download the 21 tar.gz archives from Amazon S3 as follows:

```bash
SHARD_DIR={DESIRED PATH FOR TAR.GZ files}
s5cmd --no-sign-request cp s3://argoai-argoverse/av2/tars/tbv/*.tar.gz ${SHARD_DIR}
```

If you would prefer to not install a 3rd party download tool (`s5cmd`), you can use `wget` to download the `tar.gz` files:
```bash
wget https://s3.amazonaws.com/argoai-argoverse/av2/tars/tbv/TbV_v1.0_shard0.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/av2/tars/tbv/TbV_v1.0_shard1.tar.gz
...
wget https://s3.amazonaws.com/argoai-argoverse/av2/tars/tbv/TbV_v1.0_shard20.tar.gz
```

Next, extract TbV tar.gz files that were just downloaded to a local disk using [`untar_tbv.py`](https://github.com/argoai/av2-api/blob/main/tutorials/untar_tbv.py):
```bash
python tutorials/untar_tbv.py
```
**Not Recommended**: If you want to directly transfer the extracted files, you may use:
```bash
DESIRED_TBV_DATAROOT={DESIRED LOCAL DIRECTORY PATH FOR TBV VEHICLE LOGS}
s5cmd --no-sign-request cp s3://argoai-argoverse/av2/tbv/* ${DESIRED_TBV_DATAROOT}
```

## Log Distribution Across Cities
TbV vehicle logs are captured in 6 cities, according to the following distribution:
- Austin, Texas: 80 logs.
- Detroit, Michigan: 139 logs.
- Miami, Florida: 349 logs.
- Pittsburgh, Pennsylvania: 318 logs.
- Palo Alto, California: 21 logs.
- Washington, D.C.: 136 logs.


## Baselines
We provide both pre-trained models for HD map change detection and code for training such models at [https://github.com/johnwlambert/tbv](https://github.com/johnwlambert/tbv).

## Sensor Suite

The sensor suite is identical to the Argoverse 2 Sensor Dataset, except no stereo sensor data is provided, and the sensor imagery for 6 of the cameras is provided at half of the image resolution (`ring_front_center` is at an identical resolution, however).

Lidar sweeps are collected at 10 Hz, along with 20 fps imagery from 7 ring cameras positioned to provide a fully panoramic field of view. In addition, camera intrinsics, extrinsics and 6-DOF ego-vehicle pose in a global coordinate system are provided. Lidar returns are captured by two 32-beam lidars, spinning at 10 Hz in the same direction, but separated in orientation by 180°. The cameras trigger in-sync with both lidars, leading to a 20 Hz frame-rate. The seven global shutter cameras are synchronized to the lidar to have their exposure centered on the lidar sweeping through their fields of view. 

We aggregate all returns from the two stacked 32-beam sensors into a single sweep. These sensors each have different, overlapping fields-of-view. Both lidars have their own reference frame, and we refer to them as `up_lidar` and `down_lidar`, respectively. We have egomotion-compensated the LiDAR sensor data to the egovehicle reference nanosecond timestamp. **All LiDAR returns are provided in the egovehicle reference frame, not the individual LiDAR reference frame**.

TbV imagery is provided at (height x width) of  `2048 x 1550` (portrait orientation) for the ring front-center camera, and at `775 x 1024` (landscape orientation) for all other 6 cameras. Please note that the ring front-center camera imagery is provided at higher resolution. **All camera imagery is provided in an undistorted format**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158674314-f0c930a7-8f46-42e7-b849-3bbfd72b63db.png" height="400">
</p>

## Dataset Structure Format

Tabular data (lidar sweeps, poses, calibration) are provided as [Apache Feather Files](https://arrow.apache.org/docs/python/feather.html) with the file extension `.feather`. We show examples below.

Unlike the Argoverse 2 Sensor Dataset, TbV features no object annotations.

## Maps

A local vector map and a local ground height raster map is provided per log, please refer to the [Map README](../../map/README.md) for additional details. For example, for log `VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020`, the `map` subdirectory contains 3 files:

- `log_map_archive_VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020____DTW_city_73942.json`: local vector map.
- `VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020_ground_height_surface____DTW.npy`: local ground height raster map, at 30 cm resolution.
- `VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020___img_Sim2_city.json`: mapping from city coordinates to raster grid/array coordinates.

## Pose

6-DOF ego-vehicle pose in a global (city) coordinate system is provided (visualized in the figure below as a red line, with red circles indicated at a 1 Hz frequency):
<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/157916600-cd67a529-596e-4a75-bb99-d69bbfb9075b.jpg" height="800">
</p>

We refer to this pose as `city_SE3_egovehicle` throughout the codebase:

```python
>>> import av2.utils.io as io_utils
>>> io_utils.read_feather("{TBV_ROOT}/VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020/city_SE3_egovehicle.feather")
            timestamp_ns        qw        qx        qy        qz         tx_m         ty_m       tz_m
0     315969466027482498  0.245655  0.009583 -0.014121 -0.969207  9277.579933  6805.407468 -22.647127
1     315969466042441191  0.245661  0.009824 -0.014529 -0.969197  9277.496340  6805.362364 -22.647355
2     315969466057428264  0.245682  0.009999 -0.015003 -0.969183  9277.418457  6805.317208 -22.648150
3     315969466060265000  0.245687  0.010025 -0.015133 -0.969179  9277.402699  6805.308645 -22.648235
4     315969466077482496  0.245723  0.010218 -0.015682 -0.969159  9277.306645  6805.257303 -22.648716
...                  ...       ...       ...       ...       ...          ...          ...        ...
8811  315969525887425441  0.843540  0.008404 -0.005364 -0.536974  9371.218847  6465.181151 -23.095571
8812  315969525892441193  0.843547  0.008349 -0.005421 -0.536963  9371.243129  6465.129394 -23.097279
8813  315969525899927216  0.843569  0.008234 -0.005435 -0.536930  9371.278003  6465.054774 -23.097989
8814  315969525907428274  0.843575  0.008092 -0.005358 -0.536924  9371.312815  6464.980204 -23.098440
8815  315969525912451243  0.843601  0.008013 -0.005400 -0.536883  9371.333643  6464.934933 -23.095809

[8816 rows x 8 columns]
```

## LiDAR Sweeps

For example, we show below the format of an example sweep `sensors/lidar/315969468259945000.feather` (the sweep has a reference timestamp of 315969468259945000 nanoseconds). Unlike the sensor dataset, TbV sweeps **do not** contain timestamps per return (there is no `offset_ns` attribute):

```python
>>> io_utils.read_feather("{TBV_ROOT}/VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020/sensors/lidar/315969468259945000.feather")
               x          y          z  intensity  laser_number
0     -13.023438  12.492188  -0.138794        103            25
1     -10.992188  10.726562   1.831055         36             7
2     -15.273438  14.460938   0.356445         35            23
3     -10.828125  10.609375   1.076172         49            19
4     -10.570312  10.421875   1.456055        104             3
...          ...        ...        ...        ...           ...
89261   4.136719  -2.882812   1.631836          0            19
89262   4.054688  -2.783203   1.546875         23             3
89263  60.312500 -77.937500  10.671875         47            25
89264  17.984375 -21.390625   1.214844          6             7
89265   4.160156  -2.953125   1.719727         36            23

[89266 rows x 5 columns]
```

## Calibration

An example calibration file is shown below, parameterizing `vehicle_SE3_sensor` for each sensor (the sensor's pose in the egovehicle coordinate system):

```python
>>> io_utils.read_feather("{TBV_ROOT}/VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020/calibration/egovehicle_SE3_sensor.feather")
         sensor_name        qw        qx        qy        qz      tx_m      ty_m      tz_m
0  ring_front_center  0.501067 -0.499697  0.501032 -0.498200  1.626286 -0.020252  1.395709
1    ring_front_left  0.635731 -0.671186  0.277021 -0.261946  1.549577  0.177582  1.388212
2   ring_front_right  0.262148 -0.277680  0.670922 -0.635638  1.546437 -0.216452  1.394248
3     ring_rear_left  0.602832 -0.602666 -0.368113  0.371322  1.099130  0.106534  1.389519
4    ring_rear_right  0.371203 -0.367863 -0.601619  0.604103  1.101165 -0.141049  1.399768
5     ring_side_left  0.686808 -0.722414 -0.058060  0.055145  1.308706  0.255756  1.379285
6    ring_side_right  0.055626 -0.056105 -0.722917  0.686403  1.306407 -0.291250  1.394200
7           up_lidar  0.999995  0.000000  0.000000 -0.003215  1.350110 -0.013707  1.640420
8         down_lidar  0.000080 -0.994577  0.103998  0.000039  1.355172 -0.021696  1.507259
```

## Intrinsics

An example camera intrinsics file is shown below:

```python
>>> io_utils.read_feather("{TBV_ROOT}/VvgE5LfOzIahbS266MFW7tP2al00LhQn__Autumn_2020/calibration/intrinsics.feather")
         sensor_name        fx_px        fy_px       cx_px        cy_px        k1        k2        k3  height_px  width_px
0  ring_front_center  1686.020228  1686.020228  775.467979  1020.785939 -0.245028 -0.196287  0.301861       2048      1550
1    ring_front_left   842.323546   842.323546  513.397368   387.828521 -0.262302 -0.108561  0.179488        775      1024
2   ring_front_right   842.813516   842.813516  514.154170   387.181497 -0.257722 -0.125524  0.199077        775      1024
3     ring_rear_left   841.669682   841.669682  513.211190   387.324359 -0.257018 -0.130649  0.204405        775      1024
4    ring_rear_right   843.832813   843.832813  512.201788   387.673600 -0.256830 -0.132244  0.208272        775      1024
5     ring_side_left   842.178507   842.178507  512.314602   388.188297 -0.256152 -0.131642  0.205564        775      1024
6    ring_side_right   842.703781   842.703781  513.191605   386.876520 -0.260558 -0.110271  0.179140        775      1024
```

## Privacy

All faces and license plates, whether inside vehicles or outside of the drivable area, are blurred extensively to preserve privacy.
