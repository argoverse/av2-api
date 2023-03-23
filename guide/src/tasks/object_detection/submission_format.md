# 3D Object Detection Submission Format

The evaluation expects the following fields within a `pandas.DataFrame`:

- `tx_m`: x-component of the object translation in the egovehicle reference frame.
- `ty_m`: y-component of the object translation in the egovehicle reference frame.
- `tz_m`: z-component of the object translation in the egovehicle reference frame.
- `length_m`: Object extent along the x-axis in meters.
- `width_m`: Object extent along the y-axis in meters.
- `height_m`: Object extent along the z-axis in meters.
- `qw`: Real quaternion coefficient.
- `qx`: First quaternion coefficient.
- `qy`: Second quaternion coefficient.
- `qz`: Third quaternion coefficient.
- `score`: Object confidence.
- `log_id`: Log id associated with the detection.
- `timestamp_ns`: Timestamp associated with the detection.
- `category`: Object category.

An example looks like this:

```python
# These detections are only for example purposes.

display(detections)  # Detections is type `pd.DataFrame`
                tx_m       ty_m      tz_m  length_m   width_m  height_m        qw   qx   qy        qz     score                                log_id        timestamp_ns         category
0        -162.932968   1.720428  0.039064  1.596262  0.772320  1.153996  0.125843  0.0  0.0  0.992050  0.127634  b0116f1c-f88f-3c09-b4bf-fc3c8ebeda56  315968193659921000       WHEELCHAIR
1        -120.362213  19.875946 -0.382618  1.441901  0.593825  1.199819  0.802836  0.0  0.0  0.596200  0.126565  b0116f1c-f88f-3c09-b4bf-fc3c8ebeda56  315968193659921000          BICYCLE
...
14000000   10.182907  29.489899  0.662969  9.166531  1.761454  1.615999  0.023469  0.0  0.0 -0.999725  0.322177  b2d9d8a5-847b-3c3b-aed1-c414319d20af  315978610360111000  REGULAR_VEHICLE

detections.columns
Index(['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx',
       'qy', 'qz', 'score', 'log_id', 'timestamp_ns', 'category'],
      dtype='object')
```

We need to export the above dataframe for submission. This can be done by:

```python
import pandas as pd

detections.to_feather("detections.feather")
```

Lastly, submit this file to the competition leaderboard.