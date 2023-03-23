# Introduction

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
