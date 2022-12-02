# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with the exception that v0.X updates include backwards-incompatible API changes.
From v1.0.0 and on, the project will adherence strictly to Semantic Versioning.


## [0.18.0]

### Added
- ONNX compatibility to all library modules!
- `co.onnx.export` function mirroring `torch.onnx.export`.
- purely functional `_forward_step` function to all modules.
- `_state_shape` and `_dynamic_state_inds` properties to modules.
- Add about info to package root.

### Changed
- Change call_mode internally from enum to tensor.
- Change internal state_indexes to tensors.
- Change stride to tuple.
- Change padding to tuple.

### Fixed 
- Fix assertion bug in co.Lambda.

## Removed
- TensorPlaceholder in favour of `None`.


## [0.17.1]

### Added
- Missing part on Continual Transformers in README.

### Removed
- Conv cpp impl.


## [0.17.0]

### Added
- Citations for Continual Inference lib paper.
- Docs.
- Automatic conversion for RNN modules.
- Continual Transformer modules, including:
    - `RecyclingPositionalEncoding`
    - `RetroactiveMultiheadAttention`
    - `SingleOutputMultiheadAttention`
    - `SingleOutputTransformerEncoderLayer`
    - `RetroactiveTransformerEncoderLayer`
    - `TransformerEncoderLayerFactory`
    - `TransformerEncoder`


## [0.16.0]

### Added
- "lagging" option for shrink in `co.Delay` and `co.Residual`.
- `co.RNN`.
- `co.LSTM`.
- `co.GRU`.

### Changed
- `phantom_padding` renamed to `residual_shrink`.


## [0.15.6]
### Fixed
- Missing cpp file in package.


## [0.15.5]
### Added
- CoConv step impl in C++.

## [0.15.4]
### Fixed
- FLOPs module registration compatibility with ptflops >=v0.6.8.


## [0.15.3]
### Added
- Call-mode specific functions in `co.Lambda`


## [0.15.2]
### Added
- Support for functor in `co.Lambda`

### Remove
- `nn.LayerNorm` from automatically convertible modules


## [0.15.1]
### Added
- `nn.LayerNorm` to automatically convertible modules


## [0.15.0]
### Added
- `ParallelDispatch` module.
- `Conditional` predicate print in `__repr__`.

### Fixed
- Sequential `padding` computation.
- `Lambda` `__repr__` function prints.

### Removed
- CI testing for python v3.6.


## [0.14.0]
### Added
- Added `phantom_padding` to `Residual`.
- Added `receptive_field` property.
- Added `Reshape` module.

### Changed
- Rename `forward_shrink` argument to `auto_shrink` in `Delay`.
- Torch requirement to v1.9.
- Replace `Lambda` unsqueeze_step with takes_time and new default to False.

### Fixed
- `padding` property in sequence.
- `delay` property in sequence.
- `strict` mode in `load_state_dict`.

### Removed
- Assertion error in `BroadcastReduce` for modules with different delays.


## [0.13.0]
### Added
- Add `forward_shrink` option to `Delay` and `Residual`.


## [0.12.0]
### Added
- Add `Constant`.
- Add `Zero`.
- Add `One`.


## [0.11.4]
### Fixed
- `co.ConvXd` cuda compatibility.


## [0.11.3]
### Added
- Add `flatten_state_dict` state variable.

### Removed
- Debug message for Convolutions with padding.

## [0.11.2]
### Fixed
- `call_mode` for `Linear`.


## [0.11.1]
### Added
- Add `call_mode`.
- Add `warm_up`.

### Changed
- Container implementations to use `__call__` with alternating call_modes. This change was necessary to propperly trigger the torch hooks needed in `ptflops`.

### Fixed
- `ptflops` compatibility.


## [0.11.0]
### Added
- `co.Linear` module.
- Improved repr in `co.Lambda`.
- Option to skip unsqueeze in `co.Lambda.forward_step`.


## [0.10.0]
### Changed
- Renamed `co.Parallel` to `co.BroadcastReduce`.

### Added
- `co.Broadcast` module.
- new `co.Parallel` module.
- `co.Reduce` module.
- Automatic inference of `co.Broadcast.num_streams` in `co.Sequential`.


## [0.9.0]
### Added
- `co.Lambda` module.
- `co.Add` module.
- `co.Multiply` module.
- `co.Unity` module.
- `co.Conditional` module.


## [0.8.1]
### Fixed
- Bug in `forward_stepping`.
- Bug in `clean_state`.


## [0.8.0]
### Fixed
- Bugs in `forward_step(s)` with `update_state=False`.

### Changed
- `forward_steps` interface to always include `pad_end` argument.
- Name of "interface.py" to "module.py".
- Implementations of `forward_step(s)` to be consolidated in CoModule.

### Removed
- `Padded` interface.


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
- `co._PoolNd` attribute naming: "temporal_*" removed as prefix for kernel_size, stride, dilation, and padding.


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
