# Forecasting Submission Format

The evaluation expects a dictionary of dictionaries of lists of dictionaries

```python
{
  <log_id>: {
    <timestamp_ns>: [
      {
         "prediction_m": <prediction>
         "score": <score>
         "detection_score": <detection_score>,
         "instance_id": <instance_id>
         "current_translation_m": <current_translation_m>,
         "label": <label>,
         "name": <name>,
         "size": <size>,
         "yaw" : <yaw>
      }, ...
    ], ...
  }, ...
}
```

- `log_id`: Log id associated with the forecast, also called `seq_id`.
- `timestamp_ns`: Timestamp associated with the detections.
- `prediction_m`: K translation forecasts 3 seconds into the future.
- `score`: Forecast confidence.
- `detection_score`: Detection confidence.
- `instance_id`: Unique id assigned to each object.
- `current_translation_m`: xyz-components of the object translation in the city reference frame at the current timestamp_ns, in meters.
- `label`: Integer index of the object class.
- `name`: Object class name.
- `size`: Object extent along the x,y,z axes in meters.
- `yaw` : Object rotation in radians

An example looks like this:

```python
# These forecasts are only for example purposes.

print(forecasts)
{
  '02678d04-cc9f-3148-9f95-1ba66347dff9': {
    315969904359876000: [
      {'timestep_ns': 315969905359854000,
      'current_translation_m': array([6759.4230302 , 1596.38016309]),
      'detection_score': 0.54183,
      'size': array([4.4779487, 1.7388916, 1.6963532], dtype=float32),
      'yaw' : 0.832,
      'label': 0,
      'name': 'REGULAR_VEHICLE',
      'prediction_m': array([[[6759.4230302 , 1596.38016309],
              [6759.42134062, 1596.38361481],
              [6759.41965104, 1596.38706653],
              [6759.41796145, 1596.39051825],
              [6759.41627187, 1596.39396997],
              [6759.41458229, 1596.39742169]],
 
              [[6759.4230302 , 1596.38016309],
              [6759.4210027 , 1596.38430516],
              [6759.4189752 , 1596.38844722],
              [6759.4169477 , 1596.39258928],
              [6759.4149202 , 1596.39673134],
              [6759.41289271, 1596.40087341]],
 
              [[6759.4230302 , 1596.38016309],
              [6759.42066479, 1596.3849955 ],
              [6759.41829937, 1596.38982791],
              [6759.41593395, 1596.39466031],
              [6759.41356854, 1596.39949272],
      ...
              [6759.41998895, 1596.38637619],
              [6759.4189752 , 1596.38844722],
              [6759.41796145, 1596.39051825]]]),
      'score': [0.54183, 0.54183, 0.54183, 0.54183, 0.54183],
      'instance_id': 0},
      ...
    ]
    ...
  }
}
```

We need to export the above dictionary for submission. This can be done by:

```python
import pickle

with open("forecast_predictions.pkl", "wb") as f:
       pickle.dump(forecasts, f)
```

Lastly, submit this file to the competition leaderboard.
