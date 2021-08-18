# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.3.0]
## Added
- Support for dilation and stride in pooling

## Changed
- Pooling API to match torch.nn better
- `_ConvCoNd.forward_steps` doesn't invoke `clean_state` anymore


## [0.2.2]
## Added
- Automatic conversion of batch normalisation and activation functions

## Fixed
- Separate dilation and stride in pool

## Changed
- Conv forward to use temporal padding like (like nn.Conv)

## Removed
- `co.BatchNorm2d`

## [0.2.1]
## Changed
- Renamed `unsqueezed` to `forward_stepping`.

## Removed 
- Unused utility `Zeros`


## [0.2.0]
## Changed
- Naming to match `torch.nn`. This lets the continual modules be used as drop-in replacements for `torch.nn` modules.
- Renamed `forward_regular_unrolled` to `forward`, `forward_regular` to `forward_steps`, and `forward` for `forward_step`.
- Renamed `from_regular` to `build_from`.
- Renamed `continual` to `unsqueezed`.

## Added
- `Sequential` wrapper for sequential application of modules
- `Parallel` wrapper for parallel application and aggregation of inputs
- `Residual` wrapper for adding a unity residual to a module
- `continual` conversion function
- `register` function for 3rd party modules to register their conversion
- Additional tests

## [0.1.2] - 2021-08-1
### Added
- Pooling modules: `MaxPool1d`, `AvgPool3d`, `MaxPool3d`, `AdaptiveAvgPool3d`, `AdaptiveMaxPool3d`.


## [0.1.1] - 2021-08-10
### Added
- Updated README.


## [0.1.0] - 2021-08-10
### Added
- Initial publicly available implementation of the library.
