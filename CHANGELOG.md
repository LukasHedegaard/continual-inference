# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]
## Changed
- Naming to match `torch.nn`. This lets the continual modules be used as drop-in replacements for `torch.nn` modules.
- Renamed `forward_regular_unrolled` to `forward`, `forward_regular` to `forward_steps`, and `forward` for `forward_step`.

## [0.1.2] - 2021-08-1
### Added
- Pooling modules: `MaxPool1d`, `AvgPool3d`, `MaxPool3d`, `AdaptiveAvgPool3d`, `AdaptiveMaxPool3d`


## [0.1.1] - 2021-08-10
### Added
- Updated README


## [0.1.0] - 2021-08-10
### Added
- Initial publicly available implementation of the library
