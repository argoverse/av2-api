# Downloading the data

Our datasets are available for download from [AWS S3](https://aws.amazon.com/s3/). For the best experience, we highly recommend using the open-source [s5cmd](https://github.com/peak/s5cmd) tool to transfer the data to your local filesystem (additional info available [here](https://aws.amazon.com/blogs/opensource/parallelizing-s3-workloads-s5cmd/)). Please note that an AWS account is not required to download the datasets.

## Installing `s5cmd`

### Conda Installation (Recommended)

The easiest way to install `s5cmd` is through `conda` using the `conda-forge` channel:

```terminal
conda install s5cmd -c conda-forge
```

### Manual Installation

`s5cmd` can be easily installed with the following script:

```bash
#!/usr/bin/env bash

export INSTALL_DIR=$HOME/.local/bin
export PATH=$PATH:$INSTALL_DIR
export S5CMD_URI=https://github.com/peak/s5cmd/releases/download/v1.4.0/s5cmd_1.4.0_$(uname | sed 's/Darwin/macOS/g')-64bit.tar.gz

mkdir -p $INSTALL_DIR
curl -sL $S5CMD_URI | tar -C $INSTALL_DIR -xvzf - s5cmd
```

Note that it will install `s5cmd` in your local bin directory. You can always change the path if you prefer installing it in another directory.

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
