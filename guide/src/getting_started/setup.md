# Installation

We _highly_ recommend using `conda` with the `conda-forge` channel for package management.

## Install `conda`

You will need to install `conda` on your machine. We recommend to install the `conda-forge` version of `conda` found at https://github.com/conda-forge/miniforge#install.

## Install `av2`

Simply run:

```bash
bash install.sh
```

which will install _all_ of the necessary dependencies in a conda environment named `av2`.

To activate your environment (i.e., update your system paths), run:

```bash
conda activate av2
```

## FAQ

> Why manage dependencies in `conda` instead of `pip`?

`conda` enables package management outside of the `python` ecosystem. This enables us to specify all necessary dependencies in `environment.yml`. Further, gpu-based packages (e.g., `torch`) are handled better through `conda`.

> Why `conda-forge`?

`conda-forge` is a community-driven channel of conda recipes. It includes a large number of packages which can all be properly tracked in the `conda` resolver allowing for consistent environments without conflicts.

# Downloading the Argoverse 2 Datasets

Our datasets are available for download from [AWS S3](https://aws.amazon.com/s3/). For the best experience, we highly recommend using the open-source [s5cmd](https://github.com/peak/s5cmd) tool to transfer the data to your local filesystem (additional info available [here](https://aws.amazon.com/blogs/opensource/parallelizing-s3-workloads-s5cmd/)). Please note that an AWS account is not required to download the datasets.

## Installing `s5cmd`

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
