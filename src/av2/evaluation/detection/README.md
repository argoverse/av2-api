# 3D Object Detection Evaluation

Welcome to the 3D object detection API!

TL;DR

- **26** evaluation categories.
- **200 m** range (Euclidean distance in 3D).
- **CDS** is the leaderboard ranking metric.

## Overview

Our 3D object detection challenge is comprised of _26_ different annotation categories.

## Evaluation

- **Average Precision (AP)**: Standard VOC-style average precision calculation except a true positive requires a _3D_ Euclidean distance of less than a predefined threshold.

- **Average Translation Error (ATE)**: The average _3D_ Euclidean distance (center-based) between a detection and its ground truth assignment.

- **Average Scale Error (ASE)**: The average intersection over union (IoU) after the prediction and assigned ground truth's pose has been aligned.

- **Average Orientation Error (AOE)**: The average angular distance between the detection and the assigned ground truth. We choose the smallest angle between the two different headings when calculating the error.

- **Composite Detection Score (CDS)** [^1]: The ranking metric for the detection leaderboard. This is computed as:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math={\text{CDS} = \text{mAP} \cdot \sum_{x \in \mathcal{X}} 1 - x \quad \text{where} \quad  \mathcal{X} = \{ \text{mATE}_{\text{unit}}, \text{mASE}, \text{mAOE}_{\text{unit}} \}}">
</p>


<!-- $$\text{CDS} = \text{mAP}  \sum_{x \in \mathcal{X}} 1 - x \quad \text{where} \quad  \mathcal{X} = \{ \text{mATE}_{\text{unit}}, \text{mASE}, \text{mAOE}_{\text{unit}} \}.$$ [^1] -->

[^1]: We refer to metrics which aren’t upper bounded by 1 (e.g., ATE and AOE) as their normalized variants: ATE<sub>unit</sub>, AOE<sub>unit</sub>.
