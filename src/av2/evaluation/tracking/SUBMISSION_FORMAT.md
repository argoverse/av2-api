# Tracking Submission Format

The evaluation expects a dictionary of lists of dictionaries

```python
{
      <log_id>: [
            {
                  "timestamp_ns": <timestamp_ns>,
                  "track_id": <track_id>
                  "score": <score>,
                  "label": <label>,
                  "name": <name>,
                  "translation_m": <translation_m>,
                  "size": <size>,
                  "yaw": <yaw>,
                  "velocity": <velocity>,
            }
      ]
}
```

- `log_id`: Log id associated with the track, also called `seq_id`.
- `timestamp_ns`: Timestamp associated with the detections.
- `track_id`: Unique id assigned to each track, this is produced by your tracker.
- `score`: Track confidence.
- `label`: Integer index of the object class.
- `name`: Object class name.
- `translation_m`: xyz-components of the object translation in the city reference frame, in meters.
- `size`: Object extent along the x,y,z axes in meters.
- `yaw`: Object heading rotation along the z axis.
- `velocity`: Object veloicty along the x,y,z axes.

An example looks like this:

```python
# These tracks are only for example purposes.

print(tracks)
{
  '02678d04-cc9f-3148-9f95-1ba66347dff9': [
    {
       'timestamp_ns': 315969904359876000,
       'translation_m': array([[6759.51786422, 1596.42662849,   57.90987307],
             [6757.01580393, 1601.80434654,   58.06088218],
             [6761.8232099 , 1591.6432147 ,   57.66341136],
             ...,
             [6735.5776378 , 1626.72694938,   59.12224152],
             [6790.59603472, 1558.0159741 ,   55.68706682],
             [6774.78130127, 1547.73853494,   56.55294184]]),
       'size': array([[4.315736  , 1.7214599 , 1.4757565 ],
             [4.3870926 , 1.7566483 , 1.4416479 ],
             [4.4788623 , 1.7604711 , 1.4735452 ],
             ...,
             [1.6218852 , 0.82648355, 1.6104599 ],
             [1.4323177 , 0.79862624, 1.5229694 ],
             [0.7979312 , 0.6317313 , 1.4602867 ]], dtype=float32),
      'yaw': array([-1.1205611 , ... , -1.1305285 , -1.1272993], dtype=float32),
      'velocity': array([[ 2.82435445e-03, -8.80148250e-04, -1.52388044e-04],
             [ 1.73744695e-01, -3.48345393e-01, -1.52417628e-02],
             [ 7.38469649e-02, -1.16846527e-01, -5.85577238e-03],
             ...,
             [-1.38887463e+00,  3.96778419e+00,  1.45435923e-01],
             [ 2.23189720e+00, -5.40360805e+00, -2.14317040e-01],
             [ 9.81130002e-02, -2.00860636e-01, -8.68975817e-03]]),
      'label': array([ 0, 0, ... 9,  0], dtype=int32),
      'name': array(['REGULAR_VEHICLE', ..., 'STOP_SIGN', 'REGULAR_VEHICLE'], dtype='<U31'),
      'score': array([0.54183, ..., 0.47720736, 0.4853499], dtype=float32),
      'track_id': array([0, ... , 11, 12], dtype=int32),
    },
    ...
  ],
  ...
}
```

We need to export the above dictionary for submission. This can be done by:

```python
import pickle

with open("track_predictions.pkl", "wb") as f:
       pickle.dump(tracks, f)
```

Lastly, submit this file to the competition leaderboard.
