# 3D Object Detection

## Table of Contents

<!-- toc -->

## Overview

The Argoverse 3D Object Detection task differentiates itself with its **26** category taxonomy and **long-range** (150 m) detection evaluation. We detail the task, metrics, evaluation protocol, and detailed object taxonomy information below.


## Task Definition

For a unique tuple, `(log_id, timestamp_ns)`, produce a _ranked_ set of predictions $\mathcal{P}$ that describe an object's location, size, and orientation in the 3D scene:

$$
\begin{align}
    \mathcal{P} &= \left\{ x^{i}_{\text{ego}}, y^{i}_{\text{ego}}, z^{i}_{\text{ego}}, l^{i}_{\text{obj}}, w^{i}_{\text{obj}}, h^{i}_{\text{obj}}, \theta^{i}_{\text{obj}}, c^i, o^i \right\}_{i=1}^{N} \quad \text{where}, \\\\

    x^{i}_{\text{ego}} &: \text{Location along the x-axis in the ego-vehicle reference frame.} \\
    y^{i}_{\text{ego}} &: \text{Location along the y-axis in the ego-vehicle reference frame.} \\
    z^{i}_{\text{ego}} &: \text{Location along the z-axis in the ego-vehicle reference frame.} \\
    l^{i}_{\text{obj}} &: \text{Extent along the x-axis in the object reference frame.} \\
    w^{i}_{\text{obj}} &: \text{Extent along the y-axis in the object reference frame.} \\
    h^{i}_{\text{obj}} &: \text{Extent along the z-axis in the object reference frame.} \\
    \theta^{i}_{\text{obj}} &: \text{Counter clockwise rotation from the x-axis in the object reference frame.} \\
    c^{i} &: \text{Predicted likelihood.} \\
    o^{i} &: \text{Categorical object label.}
\end{align}
$$

# 3D Object Detection Taxonomy

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

## Metrics

All of our reported metrics require _assigning_ predictions to ground truth annotations written as $a_{\text{pd}, \text{gt}}$ to compute true positives (TP), false positives (FP), and false negatives (FN).

Formally, we define a _true positive_ as:

$$
\text{TP}_{\text{pd}, \text{gt}} = \left\{ a_{\text{pd}, \text{gt}} : \lVert v_{\text{pd}} - v_{\text{gt}} \rVert_2 \leq d \right\},
$$

where $d$ is a distance threshold in meters.

```admonish important
Duplicate assignments are considered _false positives_. 
```

### Average Precision

Average precision measures the area underneath the precision / recall curve across different true positive distance thresholds.

$$
\begin{align}
    \text{AP} &= \frac{1}{100}\underset{d \in \mathcal{D}}{\sum}\underset{r\in\mathcal{R}}{\sum}\text{p}_{\text{interp}}(r) \quad \text{where} \\
    \quad \mathcal{D} &= \left\{ 0.5 \text{ m}, 1.0 \text{ m}, 2.0 \text{ m}, 4.0 \text{ m} \right\} \\
    \quad \mathcal{R} &= \left\{ 0.01, 0.02, \dots, 1.00 \right\}
\end{align}
$$

### True Positive Metrics

All true positive metrics are at a $2 \text{ m}$ threshold. 

#### Average Translation Error (ATE)

ATE measures the distance between true positive assignments.

$$
\begin{align}
    \text{ATE} = \lVert t_{\text{pd}}-t_{\text{gt}} \rVert_2 \quad \text{where} \quad t_{\text{pd}}\in\mathbb{R}^3,t_{\text{gt}}\in\mathbb{R}^3.
\end{align}
$$

#### Average Scale Error (ASE)

ASE measures the shape misalignent for true positive assignments.

$$
\begin{align}
    \text{ASE} = 1 - \underset{d\in\mathcal{D}}{\prod}\frac{\min(d_{\text{pd}},d_{\text{gt}})}{\max(d_{\text{pd}},d_\text{gt})}.
\end{align}
$$

#### Average Orientation Error (AOE)

AOE measures the minimum angle between true positive assignments.

$$
\begin{align}
    \text{AOE} = |\theta_{\text{pd}}-\theta_{\text{gt}}| \quad \text{where} \quad \theta_{\text{pd}}\in[0,\pi) \text{ and } \theta_{\text{gt}}\in[0,\pi).
\end{align}
$$

### Composite Detection Score (CDS)

CDS measures the _overall_ performance across all previously introduced metrics.

$$
\begin{align}
    \text{CDS}&= \text{AP} \cdot \underset{x\in\mathcal{X}}{\sum}{ 1-x }, \\
    \mathcal{X}&=\{\text{ATE}_{\text{unit}},\text{ASE}_{\text{unit}},\text{AOE}_{\text{unit}}\},
\end{align}
$$

where $\{\text{ATE}_{\text{unit}},\text{ASE}_{\text{unit}},\text{AOE}_{\text{unit}}\}$ are the _normalized_ true positive errors.

```admonish note
$\text{ATE}$, $\text{ASE}$, and $\text{AOE}$ are bounded by $2 \text{ m}$, $1$, and $\pi$.
```

```admonish important
CDS is the **ranking** metric.
```

# Evaluation

The 3D object detection evaluation consists of the following steps:

1. Partition the predictions and ground truth objects by a unique id, `(log_id: str, timestamp_ns: uint64)`, which corresponds to a single sweep.

2. For the predictions and ground truth objects associated with a single sweep, greedily assign the predictions to the ground truth objects in _descending_ order by _likelihood_.

3. Compute the true positive, false positive, and false negatives.

4. Compute the true positive metrics.

2. True positive, false positive, and false negative computation.
