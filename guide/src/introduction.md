[![PyPI Versions](https://img.shields.io/pypi/pyversions/av2)](https://pypi.org/project/av2/)
![CI Status](https://github.com/argoai/av2-api/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# Argoverse 2 API

> _Official_ GitHub repository for the [Argoverse 2](https://www.argoverse.org) family of datasets.

## Announcements

> 05/20/22 - Argoverse competitions are live!
> - [3D Object Detection](https://eval.ai/challenge/1710/overview) (Argoverse 2)
>      - [Baseline](https://github.com/benjaminrwilson/torchbox3d)
>  - [Motion Forecasting](https://eval.ai/challenge/1719/overview) (Argoverse 2)
>  - [Stereo](https://eval.ai/challenge/1704/overview) (Argoverse 1)

## Getting Started

### Setup

The easiest way to install the API is via [conda](https://docs.conda.io/en/latest/) by running the following command:

```bash
bash conda/install.sh
```

Additional information can be found in [INSTALL.md](conda/INSTALL.md).

### Supported Datasets

<div align="center">

| Dataset | Scenarios | Cameras | Lidar| Maps|
| :---------------| :-------- | :------------ | :-- | :-- |
| [Sensor](src/av2/datasets/sensor/README.md) | `1,000` |  :ballot_box_with_check: | :ballot_box_with_check: | :ballot_box_with_check: |
| [Lidar](src/av2/datasets/lidar/README.md)  | `20,000` | | :ballot_box_with_check: | :ballot_box_with_check: |
| [Motion Forecasting](src/av2/datasets/motion_forecasting/README.md) | `250,000` | | | :ballot_box_with_check: | [Motion Forecasting Dataset README](src/av2/datasets/motion_forecasting/README.md) |
| [Map Change (Trust, but Verify)](src/av2/datasets/tbv/README.md) | `1,045` | :ballot_box_with_check:  | :ballot_box_with_check: | :ballot_box_with_check: | [Map Change Dataset README](src/av2/datasets/tbv/README.md) |
  
</div>

<div align="center">
  <h4> <a href="src/av2/datasets/sensor/README.md"> Sensor Dataset </a> </h4>
  <img src="https://user-images.githubusercontent.com/29715011/158742778-557f31a4-569d-44aa-a032-99836094dc97.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158742776-069501c4-8dd4-4f9d-ac8c-f0421f855607.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158739736-fe876299-23da-46ed-98ce-173f938d1702.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158739767-886e1c2f-4613-495d-9204-a7b4813af16d.gif" height="150">
</div>

<div align="center">
  <h4> <a href="src/av2/datasets/lidar/README.md"> Lidar Dataset </a> </h4>
  <img src="https://user-images.githubusercontent.com/29715011/158715494-472339d1-a5d5-4d33-8fcf-3455c0d78d27.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158715496-f439ccad-71af-4880-8b43-ade7b6c8f333.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158715498-23d7a11f-12a1-4aeb-b9af-dbced217b340.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158715497-d1603423-c32f-4cf0-ab1e-6bbc9c458535.gif" height="150">
</div>


<div align="center">
  <h4> <a href="src/av2/datasets/motion_forecasting/README.md"> Motion Forecasting Dataset </a> </h4>
  <img src="https://user-images.githubusercontent.com/29715011/158486284-1a0df794-ee0a-4ae6-a320-0dd0d1daad06.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158486286-e734e654-b879-4994-a129-9957cc591af4.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158486288-5e7c0971-de0c-4ff5-bea7-76f7922dd1e0.gif" height="150">
</div>

<div align="center">
  <h4> <a href="src/av2/datasets/tbv/README.md"> Map Change Dataset (Trust, but Verify) </a> </h4>
  <img src="https://user-images.githubusercontent.com/29715011/159289930-a58147c3-c6ed-4b4e-a2a8-e23c23feb43e.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/159289891-8aae12e7-136a-4f44-bbc1-8ef93f01e23e.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/159152108-3c3001fe-ec7c-48fd-8c08-4a473affb2a3.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/159152102-27c04180-9ca4-4725-be81-95ee6858d367.gif" height="150">
</div>

## Downloading the Argoverse 2 Datasets
Our datasets are available for download from [AWS S3](https://aws.amazon.com/s3/). For the best experience, we highly recommend using the open-source [s5cmd](https://github.com/peak/s5cmd) tool to transfer the data to your local filesystem (additional info available [here](https://aws.amazon.com/blogs/opensource/parallelizing-s3-workloads-s5cmd/)). Please note that an AWS account is not required to download the datasets.

### Installing `s5cmd`

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

### Downloading Datasets
Once `s5cmd` is installed installed, downloading a dataset is as easy as running the following (using the sensor dataset as an example):

```bash
s5cmd --no-sign-request cp "s3://argoai-argoverse/av2/sensor/*" target-directory
```

The command will download all S3 objects to the target directory (for example, `target-directory` can be `/home/av2/sensors/`). Given the size of the dataset, it might take a couple of hours depending on the network connectivity.

When the download is finished, the dataset is ready to use!

### Dataset S3 Locations
```bash
s3://argoai-argoverse/av2/sensor/
s3://argoai-argoverse/av2/lidar/
s3://argoai-argoverse/av2/motion-forecasting/
s3://argoai-argoverse/av2/tbv/
```

## Testing

All incoming pull requests are tested using [nox](https://nox.thea.codes/en/stable/) as
part of the CI process. This ensures that the latest version of the API is always stable on all supported platforms. You
can run the full suite of automated checks and tests locally using the following command:

```bash
nox -r
```

## Contributing

Have a cool feature you'd like to add? Found an unhandled corner case? The Argoverse team welcomes contributions from
the open source community - please open a PR using the following [template](.github/pull_request_template.md)!

## Citing

Please use the following citation when referencing the [Argoverse 2](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4734ba6f3de83d861c3176a6273cac6d-Paper-round2.pdf) _Sensor_, _Lidar_, or _Motion Forecasting_ Datasets:

```BibTeX
@INPROCEEDINGS { Argoverse2,
  author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
  title = {Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```

Use the following citation when referencing the [Argoverse 2](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6f4922f45568161a8cdf4ad2299f6d23-Paper-round2.pdf) _Map Change_ Dataset:
```BibTeX
@INPROCEEDINGS { TrustButVerify,
  author = {John Lambert and James Hays},
  title = {Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```