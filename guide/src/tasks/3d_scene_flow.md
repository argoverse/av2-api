# 3D Scene Flow

## Table of Contents

<!-- toc -->

## Overview

In Argoverse 2 the LiDAR sensor samples the geometry around the AV every 0.1s, producing a set of 3D points called a "sweep". If the world were static, two successive sweeps would represent two different samples of the same geometry. In a non-static world, however, each point measured in the first sweep could have moved before being sampled again. 3D Scene Flow estimation aims to find these motion vectors that relate two successive LiDAR sweeps.

## Labeling Procedure

Since we do not have any direct way of measuring the motion of every point in the scene, we leverage object-level tracking labels to generate piecewise-rigid flow labels. We have a set of oriented bounding boxes for each sweep, one for each annotated object. For each bounding box, if the second sweep contains a corresponding bounding box, we can extract the rigid transformation that transforms points in the first box to the second. For each point inside the bounding box, we assign it the flow vector corresponding to that rigid motion. Points not belonging to any bounding box are assigned the ego-motion as flow. For objects that only appear in one frame, we cannot compute the ground truth flow, so they are ignored for evaluation purposes but included in the input.

## Input

- Sweep 1: (N x 4) The XYZ positions of each point in the first sweep as well as the intensity of the return.
- Sweep 2: (M x 4) The same but for the second sweep.
- Ego Motion: The pose of the autonomous vehicle in the second frame relative to the first.
- Ground annotations: For each sweep, we give a binary classification indicating if the point belongs to the ground as determined by the ground height map.

## Output

The purpose of the task is to produce two outputs. As described above, the main output is an N x 3 array of motion. However, we also ask that contestants submit a binary segmentation of the scene into "Dynamic" and "Static". This prediction should label points as "Dynamic" if they move faster than 0.5m/s *in the world frame*.

# Getting Started

## Data Loading

Once the Sensor Dataset is set up (see [these instructions](https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/sensor/README.md)), you can use the `SceneFlowDataloader` to load pairs of sweeps along with all the auxiliary information (poses and ground annotations) and flow annotations. The data loader can be found in `av2.torch.data_loaders.scene_flow`, and documentation can be found in the [source code](https://github.com/argoverse/av2-api/blob/main/src/av2/torch/data_loaders/scene_flow.py).

## Evaluation Point Subset

The contest only asks for flow and dynamic segmentation predictions on a subset of the input points. Specifically, we are only interested in points that do not belong to the ground and are within a 100m x 100m box centered on the origin. We offer a utility function `compute_eval_point_mask` in `av2.evaluation.scene_flow.utils` to compute this mask, but DO NOT USE THIS TO CREATE SUBMISSION FILES. To ensure consistency, we have pre-computed the masks for submission, which can be loaded using `get_eval_point_mask`.


# Contest Submission Format

The evaluation expects a zip archive of [Apache Feather](https://arrow.apache.org/docs/python/feather.html) files --- one for each example. The unzipped directory must have the format:

```terminal
- <test_log_1>/
  - <test_timestamp_ns_1>.feather
  - <test_timestamp_ns_2>.feather
  - ...
- <test_log_2>/
- ...
```

The evaluation is run on a subset of the test set. Use the utility function `get_eval_subset` to get the `SceneFlowDataloader` indices to submit. Each feather file should contain your flow predictions for the subset of points returned by `get_eval_mask` in the format:

- `flow_tx_m` (float16): x-component of the flow (in meters) in the first sweeps' ego-vehicle reference frame.
- `flow_ty_m` (float16): y-component of the flow (in meters) in the first sweeps' ego-vehicle reference frame.
- `flow_tz_m` (float16): z-component of the flow (in meters) in the first sweeps' ego-vehicle reference frame.
- `is_dynamic` (bool): Predicted dynamic/static labels for each point. A point is considered dynamic if its ground truth flow has a $\ell^2$-norm greater than $0.05 \textit{ m}$ once ego-motion has been removed.


For example, the first log in the test set is `0c6e62d7-bdfa-3061-8d3d-03b13aa21f68`, and the first timestamp is `315971435999927221`, so there should be a folder and file in the archive of the form: `0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/315971435999927221.feather`. That file should look like this:
```python
       flow_tx_m  flow_ty_m  flow_tz_m
0      -0.699219   0.002869   0.020233
1      -0.699219   0.002790   0.020493
2      -0.699219   0.002357   0.020004
3      -0.701172   0.001650   0.013390
4      -0.699219   0.002552   0.020187
...          ...        ...        ...
68406  -0.703613  -0.001801   0.002373
68407  -0.704102  -0.000905   0.002567
68408  -0.704590  -0.001390   0.000397
68409  -0.704102  -0.001608   0.002283
68410  -0.704102  -0.001619   0.002207
```
The file `example_submission.py` contains a basic example of how to output the submission files. The script `make_submission_archive.py` will create the zip archive for you and validate the submission format. Then submit the outputted file to the competition leaderboard!

# Local Evaluation

Before evaluating on the _test_ set, you will want to evaluate your model on the _validation_ set. To do this, first run `make_mask_files.py` and `make_annotation_files.py` to create files containing the minimum ground truth flow information needed to run the evaluation. Then, once your output is saved in the feather files described above, run `eval.py` to compute all leaderboard metrics.
