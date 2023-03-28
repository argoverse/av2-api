# Sensor Dataset

<div align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158742778-557f31a4-569d-44aa-a032-99836094dc97.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/158742776-069501c4-8dd4-4f9d-ac8c-f0421f855607.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/158739736-fe876299-23da-46ed-98ce-173f938d1702.gif" height="225">
  <img src="https://user-images.githubusercontent.com/29715011/158739767-886e1c2f-4613-495d-9204-a7b4813af16d.gif" height="225">
</div>

## Table of Contents

<!-- toc -->

## Overview

The Argoverse 2 Sensor Dataset is the successor to the Argoverse 1 3D Tracking Dataset. AV2 is larger, with 1,000 scenes totalling 4.2 hours of driving data, up from 113 scenes in Argoverse 1.

The total dataset amounts to 1 TB of data in its extracted form. Each vehicle log is approximately 15 seconds in duration and 1 GB in size, including ~150 LiDAR sweeps on average, and ~300 images from each of the 9 cameras (~2700 images per log).

## Sensor Suite

Lidar sweeps are collected at 10 Hz, along with 20 fps imagery from 7 ring cameras positioned to provide a fully panoramic field of view, and 20 fps imagery from 2 stereo cameras. In addition, camera intrinsics, extrinsics and 6-DOF ego-vehicle pose in a global coordinate system are provided. Lidar returns are captured by two 32-beam lidars, spinning at 10 Hz in the same direction, but separated in orientation by 180°. The cameras trigger in-sync with both lidars, leading to a 20 Hz frame-rate. The nine global shutter cameras are synchronized to the lidar to have their exposure centered on the lidar sweeping through their fields of view.

We aggregate all returns from the two stacked 32-beam sensors into a single sweep. These sensors each have different, overlapping fields-of-view. Both lidars have their own reference frame, and we refer to them as `up_lidar` and `down_lidar`, respectively. We have egomotion-compensated the LiDAR sensor data to the egovehicle reference nanosecond timestamp. **All LiDAR returns are provided in the egovehicle reference frame, not the individual LiDAR reference frame**.

Imagery is provided at (height x width) of  `2048 x 1550` (portrait orientation) for the ring front-center camera, and at `1550 x 2048` (landscape orientation) for all other 8 cameras (including the stereo cameras). **All camera imagery is provided in an undistorted format**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158674314-f0c930a7-8f46-42e7-b849-3bbfd72b63db.png" height="400">
</p>

## Dataset Structure Format

Tabular data (annotations, lidar sweeps, poses, calibration) are provided as [Apache Feather Files](https://arrow.apache.org/docs/python/feather.html) with the file extension `.feather`. We show examples below.

## Annotations
Object annotations are provided as 3d cuboids. Their pose is provided in the egovehicle's reference frame.

```python
io_utils.read_feather("{AV2_ROOT}/01bb304d-7bd8-35f8-bbef-7086b688e35e/annotations.feather")
             timestamp_ns  track_uuid          category  length_m   width_m  height_m    qw   qx   qy    qz    tx_m   ty_m  tz_m  num_interior_pts
0      315968867659956000 022c398c...           BOLLARD  0.363046  0.222484  0.746710  0.68  0.0  0.0  0.72   25.04  -2.55  0.01                10
1      315968867659956000 12361d61...           BOLLARD  0.407004  0.206964  0.792624  0.68  0.0  0.0  0.72   34.13  -2.51 -0.05                 5
2      315968867659956000 12cac1ed...           BOLLARD  0.337859  0.227949  0.747096  0.70  0.0  0.0  0.71   21.99  -2.55  0.03                13
3      315968867659956000 173910b2...           BOLLARD  0.326865  0.204709  0.809859  0.71  0.0  0.0  0.69    3.79  -2.53  0.05                16
4      315968867659956000 23716fb2...           BOLLARD  0.336697  0.226178  0.820867  0.72  0.0  0.0  0.69    6.78  -2.52  0.04                19
...                   ...         ...               ...       ...       ...       ...   ...  ...  ...   ...     ...    ...   ...               ...
18039  315968883159714000 c48fc856...          STROLLER  0.581798  0.502284  0.991001  0.97  0.0  0.0 -0.22  -10.84  34.33  0.14                13
18040  315968883159714000 cf1c4301...             TRUCK  9.500000  3.010952  3.573860 -0.51  0.0  0.0  0.85  -26.97   0.09  1.41              1130
18041  315968883159714000 a834bc72...         TRUCK_CAB  9.359874  3.260000  4.949222  0.51  0.0  0.0  0.85  138.13  13.39  0.80                18
18042  315968883159714000 ff50196f... VEHICULAR_TRAILER  3.414590  2.658412  2.583414  0.84  0.0  0.0  0.52  -13.95   8.32  1.28               533
18043  315968883159714000 a748a5c4...    WHEELED_DEVICE  1.078700  0.479100  1.215600  0.72  0.0  0.0  0.69   19.17  -6.07  0.28                 7
```
## Pose

6-DOF ego-vehicle pose in a global (city) coordinate system is provided (visualized in the figure below as a red line, with red circles indicated at a 1 Hz frequency):
<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/157916600-cd67a529-596e-4a75-bb99-d69bbfb9075b.jpg" height="800">
</p>

We refer to this pose as `city_SE3_egovehicle` throughout the codebase:

```python
>>> io_utils.read_feather("{AV2_ROOT}/54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca/city_SE3_egovehicle.feather")
            timestamp_ns        qw        qx        qy        qz        tx_m         ty_m       tz_m
0     315968112437425433 -0.740565 -0.005635 -0.006869 -0.671926  747.405602  1275.325609 -24.255610
1     315968112442441182 -0.740385 -0.005626 -0.006911 -0.672124  747.411245  1275.385425 -24.255906
2     315968112449927216 -0.740167 -0.005545 -0.006873 -0.672365  747.419676  1275.474686 -24.256406
3     315968112449927217 -0.740167 -0.005545 -0.006873 -0.672365  747.419676  1275.474686 -24.256406
4     315968112457428271 -0.739890 -0.005492 -0.006953 -0.672669  747.428448  1275.567576 -24.258680
...                  ...       ...       ...       ...       ...         ...          ...        ...
2692  315968128362451249 -0.694376 -0.001914 -0.006371 -0.719582  740.163738  1467.061503 -24.546971
2693  315968128372412943 -0.694326 -0.001983 -0.006233 -0.719631  740.160489  1467.147020 -24.545918
2694  315968128377482496 -0.694346 -0.001896 -0.006104 -0.719613  740.158684  1467.192399 -24.546316
2695  315968128387425439 -0.694307 -0.001763 -0.005998 -0.719652  740.155543  1467.286735 -24.549918
2696  315968128392441187 -0.694287 -0.001728 -0.005945 -0.719672  740.153742  1467.331549 -24.550363

[2697 rows x 8 columns]
```

## LiDAR Sweeps

For example, we show below the format of an example sweep `sensors/lidar/15970913559644000.feather` (the sweep has a reference timestamp of 15970913559644000 nanoseconds):

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

## Calibration

An example calibration file is shown below, parameterizing `vehicle_SE3_sensor` for each sensor (the sensor's pose in the egovehicle coordinate system):

```python
>>> io_utils.read_feather(f"{AV2_ROOT}/54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca/calibration/egovehicle_SE3_sensor.feather")
           sensor_name        qw        qx        qy        qz      tx_m      ty_m      tz_m
0    ring_front_center  0.502809 -0.499689  0.500147 -0.497340  1.631216 -0.000779  1.432780
1      ring_front_left  0.635526 -0.671957  0.275463 -0.262107  1.550015  0.197539  1.431329
2     ring_front_right  0.264354 -0.278344  0.671740 -0.633567  1.554057 -0.194171  1.430575
3       ring_rear_left  0.600598 -0.603227 -0.371096  0.371061  1.104117  0.124369  1.446070
4      ring_rear_right -0.368149  0.369885  0.603626 -0.602733  1.103432 -0.128317  1.428135
5       ring_side_left  0.684152 -0.724938 -0.058345  0.054735  1.310427  0.267904  1.433233
6      ring_side_right -0.053810  0.056105  0.727113 -0.682103  1.310236 -0.273345  1.435529
7    stereo_front_left  0.500421 -0.499934  0.501241 -0.498399  1.625085  0.248148  1.222831
8   stereo_front_right  0.500885 -0.503584  0.498793 -0.496713  1.633076 -0.250872  1.222173
9             up_lidar  0.999996  0.000000  0.000000 -0.002848  1.350180  0.000000  1.640420
10          down_lidar -0.000089 -0.994497  0.104767  0.000243  1.355162  0.000133  1.565252
```

## Intrinsics

An example camera intrinsics file is shown below:

```python
>>> io_utils.read_feather("{AV2_ROOT}/54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca/calibration/intrinsics.feather")
          sensor_name        fx_px        fy_px        cx_px  ...        k2        k3  height_px  width_px
0   ring_front_center  1773.504272  1773.504272   775.826693  ... -0.212167  0.328694       2048      1550
1     ring_front_left  1682.010713  1682.010713  1025.068254  ... -0.136984  0.209330       1550      2048
2    ring_front_right  1684.834479  1684.834479  1024.373455  ... -0.133341  0.208709       1550      2048
3      ring_rear_left  1686.494558  1686.494558  1025.655905  ... -0.129761  0.202326       1550      2048
4     ring_rear_right  1683.375120  1683.375120  1024.381124  ... -0.129331  0.201599       1550      2048
5      ring_side_left  1684.902403  1684.902403  1027.822264  ... -0.124561  0.196519       1550      2048
6     ring_side_right  1682.936559  1682.936559  1024.948976  ... -0.109515  0.179383       1550      2048
7   stereo_front_left  1685.825885  1685.825885  1025.830335  ... -0.113065  0.182441       1550      2048
8  stereo_front_right  1683.137591  1683.137591  1024.612074  ... -0.127301  0.198538       1550      2048
```

A local map is provided per log, please refer to the [Map README](../../map/README.md) for additional details.

## Log Distribution Across Cities
Vehicle logs from the **AV2 Sensor Dataset** are captured in 6 cities, according to the following distribution:
- Austin, Texas: 31 logs.
- Detroit, Michigan: 117 logs.
- Miami, Florida: 354 logs.
- Pittsburgh, Pennsylvania: 350 logs.
- Palo Alto, California: 22 logs.
- Washington, D.C.: 126 logs.

## Privacy

All faces and license plates, whether inside vehicles or outside of the drivable area, are blurred extensively to preserve privacy.

## Sensor Dataset splits

We randomly partitioned 1000 logs into the following splits:

- Train (700 logs)
- Validation (150 logs)
- Test (150 logs)

## Sensor Dataset Taxonomy

The AV2 Sensor Dataset contains 10 Hz 3D cuboid annotations for objects within our 30 class taxonomy. Objects are annotated if they are within the “region of interest” (ROI) – within five meters of the mapped “driveable” area.

![cuboids-by-category](https://user-images.githubusercontent.com/29715011/158675968-3987b2bc-ed8b-4194-85ca-44298805cfb2.png)

These 30 classes are defined as follows, appearing in order of frequency:

1. `REGULAR_VEHICLE`:
Any conventionally sized passenger vehicle used for the transportation of people and cargo. This includes Cars, vans, pickup trucks, SUVs, etc.

2. ``PEDESTRIAN``:
Person that is not driving or riding in/on a vehicle. They can be walking, standing, sitting, prone, etc.

3. `BOLLARD`:
Bollards are short, sturdy posts installed in the roadway or sidewalk to control the flow of traffic. These may be temporary or permanent and are sometimes decorative.

4. `CONSTRUCTION_CONE`:
Movable traffic cone that is used to alert drivers to a hazard.  These will typically be orange and white striped and may or may not have a blinking light attached to the top.

5. `CONSTRUCTION_BARREL`:
Construction Barrel is a movable traffic barrel that is used to alert drivers to a hazard.  These will typically be orange and white striped and may or may not have a blinking light attached to the top.

6. `STOP_SIGN`:
Red octagonal traffic sign displaying the word STOP used to notify drivers that they must come to a complete stop and make sure no other road users are coming before proceeding.

7. `BICYCLE`:
Non-motorized vehicle that typically has two wheels and is propelled by human power pushing pedals in a circular motion.

8. `LARGE_VEHICLE`:
Large motorized vehicles (four wheels or more) which do not fit into any more specific subclass. Examples include extended passenger vans, fire trucks, RVs, etc.

9. `WHEELED_DEVICE`:
Objects involved in the transportation of a person and do not fit a more specific class. Examples range from skateboards, non-motorized scooters, segways, to golf-carts.

10. `BUS`:
Standard city buses designed to carry a large number of people.

11. `BOX_TRUCK`:
Chassis cab truck with an enclosed cube shaped cargo area. It should be noted that the cargo area is rigidly attached to the cab, and they do not articulate.

12. `SIGN`:
Official road signs placed by the Department of Transportation (DOT signs) which are of interest to us. This includes yield signs, speed limit signs, directional control signs, construction signs, and other signs that provide required traffic control information. Note that Stop Sign is captured separately and informative signs such as street signs, parking signs, bus stop signs, etc. are not included in this class.

13. `TRUCK`:
Vehicles that are clearly defined as a truck but does not fit into the subclasses of Box Truck or Truck Cab. Examples include common delivery vehicles (UPS, FedEx), mail trucks, garbage trucks, utility trucks, ambulances, dump trucks, etc.

14. `MOTORCYCLE`:
Motorized vehicle with two wheels where the rider straddles the engine.  These are capable of high speeds similar to a car.

15. `BICYCLIST`:
Person actively riding a bicycle, non-pedaling passengers included.

16. `VEHICULAR_TRAILER`:
Non-motorized, wheeled vehicle towed behind a motorized vehicle.

17. `TRUCK_CAB`:
Heavy truck commonly known as “Semi cab”, “Tractor”, or “Lorry”. This refers to only the front of part of an articulated tractor trailer.

18. `MOTORCYCLIST`:
Person actively riding a motorcycle or a moped, including passengers.

19. `DOG`:
Any member of the canine family.

20. `SCHOOL_BUS`:
Bus that primarily holds school children (typically yellow) and can control the flow of traffic via the use of an articulating stop sign and loading/unloading flasher lights.

21. `WHEELED_RIDER`:
Person actively riding or being carried by a wheeled device.

22. `STROLLER`:
Push-cart with wheels meant to hold a baby or toddler.

23. `ARTICULATED_BUS`:
Articulated buses perform the same function as a standard city bus, but are able to bend (articulate) towards the center. These will also have a third set of wheels not present on a typical bus.

24. `MESSAGE_BOARD_TRAILER`:
Trailer carrying a large, mounted, electronic sign to display messages. Often found around construction sites or large events.

25. `MOBILE_PEDESTRIAN_SIGN`:
Movable sign designating an area where pedestrians may cross the road.

26. `WHEELCHAIR`:
Chair fitted with wheels for use as a means of transport by a person who is unable to walk as a result of illness, injury, or disability. This includes both motorized and non-motorized wheelchairs as well as low-speed seated scooters not intended for use on the roadway.

27. `RAILED_VEHICLE`:
Any vehicle that relies on rails to move. This applies to trains, trolleys, train engines, train freight cars, train tanker cars, subways, etc.

28. `OFFICIAL_SIGNALER`:
Person with authority specifically responsible for stopping and directing vehicles through traffic.

29. `TRAFFIC_LIGHT_TRAILER`:
Mounted, portable traffic light unit commonly used in construction zones or for other temporary detours.

30. `ANIMAL`:
All recognized animals large enough to affect traffic, but that do not fit into the Cat, Dog, or Horse categories
