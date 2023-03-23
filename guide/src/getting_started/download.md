# Download the Datasets

Once `s5cmd` is installed installed, downloading a dataset is as easy as running the following (using the sensor dataset as an example):

```bash
s5cmd --no-sign-request cp "s3://argoai-argoverse/av2/sensor/*" target-directory
```

The command will download all S3 objects to the target directory (for example, `target-directory` can be `/home/av2/sensors/`). Given the size of the dataset, it might take a couple of hours depending on the network connectivity.

When the download is finished, the dataset is ready to use!

## Dataset S3 Locations
```bash
s3://argoai-argoverse/av2/sensor/
s3://argoai-argoverse/av2/lidar/
s3://argoai-argoverse/av2/motion-forecasting/
s3://argoai-argoverse/av2/tbv/
```
