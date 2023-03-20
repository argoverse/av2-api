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