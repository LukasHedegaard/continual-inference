# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Fixed
- Bugs in `forward_step(s, update_state=False)`

### Changed
- `forward_steps` interface to always include `pad_end` argument
- name of "interface.py" to "module.py"

### Removed
- `Padded` interface


## [0.7.0]
### Added
- Independent state_dict and load_state_dict functions.
- Added nonempty check for aggregation functions in Parallel.
- `update_state` argument to all `forward_step(s)` methods.
- Additional tests for edge-cases

### Changed
- Changed default pad_end value to False.

### Fixed
- Continual interface and conversion to support both class and module.
- Replicate padding in `co._ConvNd`


## [0.6.1]
### Changed
- `co.Residual` modules to be unnamed. This allows the module state dicts to be flattened.

## [0.6.0]
### Added
- Flattened state dict export and loading via a `flatten` argument. This feature improves interoperability complex modules, that were not originally constructed with the `co.Sequential` and `co.Parallel` building blocks.
- Context manager for triggering flattened state_dict export and loading.


## [0.5.0]
### Added
- Support for zero-delay in `co.Delay`
- Support for broadcasting in `co.Parallel`
- Mul (hadamark product) aggregation in `co.Parallel`
- Example of Squeeze and Excitation block

### Changed
- `co._PoolNd` attribute naming: "temporal_" removed as prefix for kernel_size, stride, dilation, and padding.


## [0.4.0]
### Added
- `co.Delay` handling for padding.
- Handling of initialisation and strides in containers

### Changed
- `co.Conv` `build_from` behavior to not change dilation and stride. Argument overload supported instead. 
- `pad_start` and `pad_end` args to convolution and pooling modules `forward_steps`.
- Behavior of modules while they initialise. Now, a TensorPlaceholder is passed for initialising steps.

### Removed
- Automatic unsqueeze in pooling.


## [0.3.1]
### Added
- Support for dropout.


## [0.3.0]
### Added
- Support for dilation and stride in pooling.

### Changed
- Pooling API to match torch.nn better.
- `_ConvCoNd.forward_steps` doesn't invoke `clean_state` anymore.


## [0.2.2]
### Added
- Automatic conversion of batch normalisation and activation functions.

### Fixed
- Separate dilation and stride in pool.

### Changed
- Conv forward to use temporal padding like (like nn.Conv).

### Removed
- `co.BatchNorm2d`

## [0.2.1]
### Changed
- Renamed `unsqueezed` to `forward_stepping`.

### Removed 
- Unused utility `Zeros`


## [0.2.0]
### Changed
- Naming to match `torch.nn`. This lets the continual modules be used as drop-in replacements for `torch.nn` modules.
- Renamed `forward_regular_unrolled` to `forward`, `forward_regular` to `forward_steps`, and `forward` for `forward_step`.
- Renamed `from_regular` to `build_from`.
- Renamed `continual` to `unsqueezed`.

### Added
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
