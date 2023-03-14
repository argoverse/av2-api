# Conda Install

We _highly_ recommend using `conda` for package management. To install `av2-api`, simply run:

```bash
bash install.sh
```

which will install _all_ of the necessary dependencies in a conda environment named `av2`.

To activate your environment (i.e., update your system paths), run:

```bash
conda activate av2
```

## FAQ

> Why `conda` instead of `pip`?

`conda` enables package management outside of the `python` ecosystem. This enables us to specify all necessary dependencies in `environment.yml`. Further, gpu-based packages (e.g., `torch`) are handled better through `conda`.