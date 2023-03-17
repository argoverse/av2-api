# 3D Scene Flow Submission Format

The evaluation expects a zip archive of pandas DataFrames stored as feather files, one for each example. The unzipped directory must have the format:
- <test_log_1>/
  - <test_timestamp_ns_1>.feather
  - <test_timestamp_ns_2>.feather
  - ...
- <test_log_2>/
- ...

The evaluation is run on a subset of the test set, use the utility function `get_eval_subset` to get the `SceneFlowDataloader` indices to submit.  Each feather file should contain your flow predictions for the subset of points returned by `get_eval_mask` in the format:

- `flow_tx_m` (float16): x-component of the flow in the first sweeps's egovehicle reference frame.
- `flow_ty_m` (float16): y-component of the flow in the first sweeps's egovehicle reference frame.
- `flow_tz_m` (float16): z-component of the flow in the first sweeps's egovehicle reference frame.
- `dynamic` (bool): Your predicted dynamic/static labels for each point. A point is considered dynamic if its ground truth flow has a norm greater then 0.05m once ego-motion has been removed.


For example the first log in the test set is `0c6e62d7-bdfa-3061-8d3d-03b13aa21f68` and the first timestamp is `315971435999927221`, so there should be a folder and file in the archive of the form: `0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/315971435999927221.feather`. That fill should look like this:
```
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
The file `example_submission.py` contains a basic example of how to output the submission files. The script `make_submission_archive.py` will create the zip archive for you and validate the submission format. Then you can submit the outputted file to the competition leaderboard!


# Local Evaluation

Before evaluating on the test set, you will want to evaluate your model on the validation set. To do this, first run `make_annotation_files.py` to create a set of files containing the minimum ground truth flow information needed to run the evaluation. Then, once you have your output saved in the feather files described above, run `eval.py` to compute all of the leaderboard metrics.
