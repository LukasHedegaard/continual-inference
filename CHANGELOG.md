# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with the exception that v0.X updates include backwards-incompatible API changes.
From v1.0.0 and on, the project will adherence strictly to Semantic Versioning.

## Unpublished


## [1.2.3] - 2023-06-16

### Fixed
- Ensure state_index remains on the same device after clean_state.


## [1.2.2] - 2023-05-24

### Fixed
- Option to use strings to specify transformer activation.


## [1.2.1] - 2023-03-24

### Added
- Onnx as dev requirement.

### Changed
- Allow torch>=2.0.


## [1.2.0] - 2023-03-16

### Added
- `Skip` module.
- "leading" mode in `Residual`.


## [1.1.3] - 2023-03-15

### Added
- Description of state handling to README.

### Fixed
- Documentation formatting for `co.Identity()` examples.
- Horovod check for newer pytorch lightning versions.


## [1.1.2] - 2023-01-13

### Added
- `query_index` argument to `SingleOutputTransformerEncoderLayer`.

### Fixed
- `Residual` centred residual and `Delay` auto_delay forward_step. 


## [1.1.1] - 2023-01-10

### Added
- Support for `GroupNorm` and `InstanceNorm`


## [1.1.0] - 2022-12-19

### Added
- `append` function to `co.Sequential`.
- Production-ready docstrings for public functions.
- reduce_max to `Reduce`.

### Changed
- Rename `Unity` to `Identity` to follow `torch.nn`.
- Major overhaul of README, improving descriptions and adding benchmark.
- Major overhaul of docs, improving descriptions and adding benchmark.
- MHA warnings to only log once.

### Removed
- Unused parameters `batch_first` and `bidirectional` for RNN, GRU, and LSTM.


## [1.0.4] - 2022-12-07

### Fixed
- `co.Conditional` onnx support for single-option config.


## [1.0.3] - 2022-12-07

### Fixed
- `co.Conditional` onnx support.


## [1.0.2] - 2022-12-06

### Fixed
- `co.Conv` onnx export for kernel_size=1.


## [1.0.1] - 2022-12-02

### Added
- Ability to access onnx from root, i.e. `co.onnx`.


## [1.0.0] - 2022-12-02

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

### Removed
- TensorPlaceholder in favour of `None`.


## [0.17.1] - 2022-06-02

### Added
- Missing part on Continual Transformers in README.

### Removed
- Conv cpp impl.


## [0.17.0] - 2022-05-12

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


## [0.16.0] - 2022-04-04

### Added
- "lagging" option for shrink in `co.Delay` and `co.Residual`.
- `co.RNN`.
- `co.LSTM`.
- `co.GRU`.

### Changed
- `phantom_padding` renamed to `residual_shrink`.


## [0.15.6] - 2022-03-18
### Fixed
- Missing cpp file in package.


## [0.15.5] - 2022-03-05
### Added
- CoConv step impl in C++.

## [0.15.4] - 2022-01-28
### Fixed
- FLOPs module registration compatibility with ptflops >=v0.6.8.


## [0.15.3] - 2021-12-13
### Added
- Call-mode specific functions in `co.Lambda`


## [0.15.2] - 2021-12-11
### Added
- Support for functor in `co.Lambda`

### Removed
- `nn.LayerNorm` from automatically convertible modules


## [0.15.1] - 2021-12-10
### Added
- `nn.LayerNorm` to automatically convertible modules


## [0.15.0] - 2021-10-29
### Added
- `ParallelDispatch` module.
- `Conditional` predicate print in `__repr__`.

### Fixed
- Sequential `padding` computation.
- `Lambda` `__repr__` function prints.

### Removed
- CI testing for python v3.6.


## [0.14.0] - 2021-09-20
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


## [0.13.0] - 2021-09-14
### Added
- Add `forward_shrink` option to `Delay` and `Residual`.


## [0.12.0] - 2021-09-14
### Added
- Add `Constant`.
- Add `Zero`.
- Add `One`.


## [0.11.4] - 2021-09-08
### Fixed
- `co.ConvXd` cuda compatibility.


## [0.11.3] - 2021-09-08
### Added
- Add `flatten_state_dict` state variable.

### Removed
- Debug message for Convolutions with padding.

## [0.11.2] - 2021-09-08
### Fixed
- `call_mode` for `Linear`.


## [0.11.1] - 2021-09-06
### Added
- Add `call_mode`.
- Add `warm_up`.

### Changed
- Container implementations to use `__call__` with alternating call_modes. This change was necessary to properly trigger the torch hooks needed in `ptflops`.

### Fixed
- `ptflops` compatibility.


## [0.11.0] - 2021-08-31
### Added
- `co.Linear` module.
- Improved repr in `co.Lambda`.
- Option to skip unsqueeze in `co.Lambda.forward_step`.


## [0.10.0] - 2021-08-27
### Changed
- Renamed `co.Parallel` to `co.BroadcastReduce`.

### Added
- `co.Broadcast` module.
- new `co.Parallel` module.
- `co.Reduce` module.
- Automatic inference of `co.Broadcast.num_streams` in `co.Sequential`.


## [0.9.0] - 2021-08-26
### Added
- `co.Lambda` module.
- `co.Add` module.
- `co.Multiply` module.
- `co.Unity` module.
- `co.Conditional` module.


## [0.8.1] - 2021-08-26
### Fixed
- Bug in `forward_stepping`.
- Bug in `clean_state`.


## [0.8.0] - 2021-08-24
### Fixed
- Bugs in `forward_step(s)` with `update_state=False`.

### Changed
- `forward_steps` interface to always include `pad_end` argument.
- Name of "interface.py" to "module.py".
- Implementations of `forward_step(s)` to be consolidated in CoModule.

### Removed
- `Padded` interface.


## [0.7.0] - 2021-08-24
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


## [0.6.1] - 2021-08-23
### Changed
- `co.Residual` modules to be unnamed. This allows the module state dicts to be flattened.

## [0.6.0] - 2021-08-23
### Added
- Flattened state dict export and loading via a `flatten` argument. This feature improves interoperability complex modules, that were not originally constructed with the `co.Sequential` and `co.Parallel` building blocks.
- Context manager for triggering flattened state_dict export and loading.


## [0.5.0] - 2021-08-20
### Added
- Support for zero-delay in `co.Delay`
- Support for broadcasting in `co.Parallel`
- Mul (hadamard product) aggregation in `co.Parallel`
- Example of Squeeze and Excitation block

### Changed
- `co._PoolNd` attribute naming: "temporal_*" removed as prefix for kernel_size, stride, dilation, and padding.


## [0.4.0] - 2021-08-19
### Added
- `co.Delay` handling for padding.
- Handling of initialization and strides in containers

### Changed
- `co.Conv` `build_from` behavior to not change dilation and stride. Argument overload supported instead. 
- `pad_start` and `pad_end` args to convolution and pooling modules `forward_steps`.
- Behavior of modules while they initialize. Now, a TensorPlaceholder is passed for initializing steps.

### Removed
- Automatic unsqueeze in pooling.


## [0.3.1] - 2021-08-18
### Added
- Support for dropout.


## [0.3.0] - 2021-08-18
### Added
- Support for dilation and stride in pooling.

### Changed
- Pooling API to match torch.nn better.
- `_ConvCoNd.forward_steps` doesn't invoke `clean_state` anymore.


## [0.2.2] - 2021-08-17
### Added
- Automatic conversion of batch normalization and activation functions.

### Fixed
- Separate dilation and stride in pool.

### Changed
- Conv forward to use temporal padding like (like nn.Conv).

### Removed
- `co.BatchNorm2d`

## [0.2.1] - 2021-08-17
### Changed
- Renamed `unsqueezed` to `forward_stepping`.

### Removed 
- Unused utility `Zeros`


## [0.2.0] - 2021-08-16
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
