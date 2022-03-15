[![PyPI Versions](https://img.shields.io/pypi/pyversions/av2)](https://pypi.org/project/av2/)
![CI Status](https://github.com/argoai/argoverse2-api/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# Argoverse 2 API

Official GitHub repository for the [Argoverse 2](https://www.argoverse.org) family of datasets.

If you have any questions or run into any problems with either the data or API, please feel free to open a [GitHub issue](https://github.com/argoai/argoverse2-api/issues)!

## Overview

- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing-automation)
- [Contributing](#contributing)
- [Citing](#citing)
- [License](#license)

## Setup

The AV2 API is officially supported for Linux and MacOS on Python versions 3.8, 3.9, and 3.10.

The easiest way to install the API is via [pip](https://pypi.org/project/av2/) by running the following command:

```bash
pip install av2
```

## Usage

### Argoverse 2 Sensor Dataset

Please refer to the [sensor dataset README](src/av2/datasets/sensor/README.md) for additional details.

### Argoverse 2 Lidar Dataset

Please refer to the [lidar dataset README](src/av2/datasets/lidar/README.md) for additional details.

### Argoverse 2 Motion Forecasting Dataset

Please refer to the [motion forecasting dataset README](src/av2/datasets/motion_forecasting/README.md) for additional details.

### Map API

Please refer to the [map README](src/av2/map/README.md) for additional details about the common format for vector and
raster maps that we employ across all AV2 datasets.

## Testing

All incoming pull requests are tested using [nox](https://nox.thea.codes/en/stable/) as
part of the CI process. This ensures that the latest version of the API is always stable on all supported platforms. You
can run the full suite of automated checks and tests locally using the following command:

```bash
nox
```

## Contributing

Have a cool feature you'd like to add? Found an unhandled corner case? The Argoverse team welcomes contributions from
the open source community - please open a PR using the following [template](.github/pull_request_template.md)!

## Citing

Please use the following citation when referencing the [Argoverse 2](https://openreview.net/pdf?id=vKQGe36av4k) Sensor, Lidar, or Motion Forecasting Datasets:

```
@INPROCEEDINGS { Argoverse2,
  author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
  title = {Argoverse 2.0: Next Generation Datasets for Self-driving Perception and Forecasting},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```

Use the following citation when referencing the [Argoverse 2](https://openreview.net/pdf?id=vKQGe36av4k) Map Change Dataset:
```
@INPROCEEDINGS { TrustButVerify,
  author = {John Lambert and James Hays},
  title = {Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```

## License

All code provided within this repository is released under the MIT license and bound by the Argoverse terms of use,
please see [LICENSE](LICENSE) and [NOTICE](NOTICE) for additional details.
