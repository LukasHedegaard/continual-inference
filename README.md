<div align="left">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/logo/logo_name.svg" width=400>
  <br>
  <br>
  <a href="https://pypi.org/project/continual-inference/">
    <img src="https://img.shields.io/pypi/pyversions/continual-inference" height="20" >
  </a>
  <a href="https://badge.fury.io/py/continual-inference">
    <img src="https://badge.fury.io/py/continual-inference.svg" height="20" >
  </a>
  <a href="https://codecov.io/gh/LukasHedegaard/continual-inference">
    <img src="https://codecov.io/gh/LukasHedegaard/continual-inference/branch/main/graph/badge.svg?token=XW1UQZSEOG"/>
  </a>
  <a href="https://www.codefactor.io/repository/github/lukashedegaard/continual-inference/overview/main">
    <img src="https://www.codefactor.io/repository/github/lukashedegaard/continual-inference/badge/main" alt="CodeFactor" />
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
  <br>
  <br>
</div>

Building blocks for Continual Inference Networks in PyTorch

## Install 
```bash
pip install continual-inference
```

## Simple example
`co` modules are weight-compatible drop-in replacement for `torch.nn`, enhanced with the capability of efficient _continual inference_:

```python3
import torch
import continual as co
                                                           
#                      B, C, T, H, W
example = torch.randn((1, 1, 5, 3, 3))

conv = co.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3))

# Same exact computation as torch.nn.Conv3d ✅
output = conv(example)

# But can also perform online inference efficiently 🚀
firsts = conv.forward_steps(example[:, :, :4])
last = conv.forward_step(example[:, :, 4])

assert torch.allclose(output[:, :, : conv.delay], firsts)
assert torch.allclose(output[:, :, conv.delay], last)
```

See the "Advanced Examples" section for additional examples..

## Continual Inference Networks (CINs)
Continual Inference Networks are a neural network subset, which can make new predictions efficiently _for each new time-step_.
They are ideal for __online detection__ and monitoring scenarios, but can also be used succesfully in offline situations.

Some example CINs and non-CINs are illustrated below to 

__CIN__:
```
   O          O          O        (output)
   ↑          ↑          ↑       
nn.LSTM    nn.LSTM    nn.LSTM     (temporal LSTM)
   ↑          ↑          ↑    
nn.Conv2D  nn.Conv2D  nn.Conv2D   (spatial 2D conv)
   ↑          ↑          ↑    
   I          I          I        (input frame)
```

Here, we see that all network-modules, which do not utilise temporal information can be used for an Continual Inference Network (e.g. `nn.Conv1d` and `nn.Conv2d` on spatial data such as an image). 
Moreover, recurrent modules (e.g. `LSTM` and `GRU`), that summarize past events in an internal state are also useable in CINs.

However, modules that operate on temporal data with the assumption that the more temporal context is available than the current frame cannot be directly applied.
One such example is the spatio-temporal `nn.Conv3d` used by many SotA video recognition models (see below)

__Not CIN__:
```
          Θ              (output)   
          ↑              
      nn.Conv3D          (spatio-temporal 3D conv)
          ↑
  -----------------      (concatenate frames to clip)
  ↑       ↑       ↑    
  I       I       I      (input frame)  
```

Sometimes, though, the computations in such modules, can be cleverly restructured to work for online inference as well! 💪 

__CIN__:
```
    O          O          Θ      (output)
    ↑          ↑          ↑    
co.Conv3d  co.Conv3d  co.Conv3d  (continual spatio-temporal 3D conv)
    ↑          ↑          ↑    
    I          I          I      (input frame)
```
Here, the `ϴ` output of the `Conv3D` and `ConvCo3D` are identical! ✨

The last conversion from a non-CIN to a CIN is possible due to a recent break-through in Online Action Detection, namely [Continual Convolutions](https://arxiv.org/abs/2106.00050).

### Continual Convolutions
Below, we see principle sketches, which compare regular and continual convolutions during online / continual inference.

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/continual/regular-convolution.png" width="500">
  <br>
  (1) <br> 
  Regular Convolution. 
	A regular temporal convolutional layer leads to redundant computations during online processing of video clips, as illustrated by the repeated convolution of inputs (green b,c,d) with a kernel (blue α,β) in the temporal dimension. Moreover, prior inputs (b,c,d) must be stored between time-steps for online processing tasks.
  <br><br>
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/continual/continual-convolution.png" width="500">
  <br>
  (2) <br>
  Continual Convolution. 
	An input (green d or e) is convolved with a kernel (blue α, β). The intermediary feature-maps corresponding to all but the last temporal position are stored, while the last feature map and prior memory are summed to produce the resulting output. For a continual stream of inputs, Continual Convolutions produce identical outputs to regular convolutions.
  <br><br>  
</div>

Comparing Figures (1) and (2), we see that Continual Convolutions get rid of computational redundancies.
This can speed up online inference greatly - for example, a Continual X3D model for Human Activity Recognition has __10× less Floating Point Operations per prediction__ than the vanilla X3D models 🚀. 

> 💡  The longer the length of the temporal sequence, the larger the savings.

For more information, we refer to the [paper on Continual Convolutions](https://arxiv.org/abs/2106.00050).


## Forward modes
The library components feature three distinct forward modes, which are handy for different situations, namely `forward`, `forward_step`, and `forward_steps`:

### `forward`
Performs a full forward computation exactly as the regular layer would.
This method is handy for effient training on clip-based data.

```
         O            (O: output)
         ↑ 
         N            (N: nework module)
         ↑ 
 -----------------    (-: aggregation)
 P   I   I   I   P    (I: input frame, P: padding)
```


### `forward_step`
Performs a forward computation for a single frame and continual states are updated accordingly. This is the mode to use for continual inference.

```
O+S O+S O+S O+S   (O: output, S: updated internal state)
 ↑   ↑   ↑   ↑ 
 N   N   N   N    (N: nework module)
 ↑   ↑   ↑   ↑ 
 I   I   I   I    (I: input frame)
```

### `forward_steps`
Performs a layer-wise forward computation using the continual module.
The computation is performed frame-by-frame and continual states are updated accordingly.
The output-input mapping the exact same as that of a regular module.
This mode is handy for initialising the network on a whole clip (multipleframes) before the `forward` is usead for continual inference. 
```
         O            (O: output)
         ↑ 
 -----------------    (-: aggregation)
 O  O+S O+S O+S  O    (O: output, S: updated internal state)
 ↑   ↑   ↑   ↑   ↑
 N   N   N   N   N    (N: nework module)
 ↑   ↑   ↑   ↑   ↑
 P   I   I   I   P    (I: input frame, P: padding)
```

## Modules
Below is a list of the included modules and utilities included in the library:

<!-- TODO: Replace with link to docs once they are set up -->
- Convolutions:
    - `co.Conv1d`
    - `co.Conv2d`
    - `co.Conv3d`

- Pooling:
    - `co.AvgPool1d`
    - `co.AvgPool2d`
    - `co.AvgPool3d`
    - `co.MaxPool1d`
    - `co.MaxPool2d`
    - `co.MaxPool3d`
    - `co.AdaptiveAvgPool1d`
    - `co.AdaptiveAvgPool2d`
    - `co.AdaptiveAvgPool3d`
    - `co.AdaptiveMaxPool1d`
    - `co.AdaptiveMaxPool2d`
    - `co.AdaptiveMaxPool3d`

- Containers
    - `co.Sequential` - Sequential wrapper for modules. This module automatically performs conversions of torch.nn modules, which are safe during continual inference. These include all batch normalisation and activation function. 
    - `co.Parallel` - Parallel wrapper for modules.
    - `co.Residual` - Residual wrapper for modules.
    - `co.Delay` - Pure delay module (e.g. needed in residuals).

- Converters
    <!-- - `co.Residual` - residual connection, which automatically adds delay if needed -->
    - `co.continual` - conversion function from `torch.nn` modules to `co` modules.
    - `co.forward_stepping` - functional wrapper, which enhances temporally local `torch.nn` modules with the forward_stepping functions.

In addition, we support interoperability with a wide range of modules from `torch.nn`:

- Activation
    - `nn.BatchNorm1d`
    - `nn.Threshold`
    - `nn.ReLU`
    - `nn.RReLU`
    - `nn.Hardtanh`
    - `nn.ReLU6`
    - `nn.Sigmoid`
    - `nn.Hardsigmoid`
    - `nn.Tanh`
    - `nn.SiLU`
    - `nn.Hardswish`
    - `nn.ELU`
    - `nn.CELU`
    - `nn.SELU`
    - `nn.GLU`
    - `nn.GELU`
    - `nn.Hardshrink`
    - `nn.LeakyReLU`
    - `nn.LogSigmoid`
    - `nn.Softplus`
    - `nn.Softshrink`
    - `nn.PReLU`
    - `nn.Softsign`
    - `nn.Tanhshrink`
    - `nn.Softmin`
    - `nn.Softmax`
    - `nn.Softmax2d`
    - `nn.LogSoftmax`

- Batch Normalisation
    - `nn.BatchNorm1d`
    - `nn.BatchNorm2d`
    - `nn.BatchNorm3d`

- Dropout
    - `nn.Dropout`
    - `nn.Dropout2d`
    - `nn.Dropout3d`
    - `nn.AlphaDropout`
    - `nn.FeatureAlphaDropout`

## Advanced examples

### Continual 3D [MBConv](https://arxiv.org/pdf/1801.04381.pdf)

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/mb_conv.png" width="150">
  <br>
  MobileNetV2 Inverted residual block. Source: https://arxiv.org/pdf/1801.04381.pdf
</div>

```python3
mb_conv = co.Residual(
    co.Sequential(
      co.Conv3d(32, 64, kernel_size=(1, 1, 1)),
      nn.BatchNorm3d(64),
      nn.ReLU6(),
      co.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=64),
      nn.ReLU6(),
      co.Conv3d(64, 32, kernel_size=(1, 1, 1)),
      nn.BatchNorm3d(32),
    )
)
```

### Continual 3D [Inception module](https://arxiv.org/pdf/1409.4842v1.pdf)

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/inception_block.png" width="450">
  <br>
  Inception module with dimension reductions. Source: https://arxiv.org/pdf/1409.4842v1.pdf
</div>

```python3
def norm_relu(module, channels):
    return co.Sequential(
        module,
        nn.BatchNorm3d(channels),
        nn.ReLU(),
    )

inception_module = co.Parallel(
    co.Conv3d(192, 64, kernel_size=1),
    co.Sequential(
        norm_relu(co.Conv3d(192, 96, kernel_size=1), 96),
        norm_relu(co.Conv3d(96, 128, kernel_size=3, padding=1), 128),
    ),
    co.Sequential(
        norm_relu(co.Conv3d(192, 16, kernel_size=1), 16),
        norm_relu(co.Conv3d(16, 32, kernel_size=3, padding=1), 32),
    ),
    co.Sequential(
        co.MaxPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
        norm_relu(co.Conv3d(192, 32, kernel_size=1), 32),
    ),
    aggregation_fn="concat",
)
```


### Continual 3D [Squeeze-and-Excitation module](https://arxiv.org/pdf/1709.01507.pdf)

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/se_block.png" width="230">
  <br>
  Squeeze-and-Excitation block. 
  Scale refers to a broadcasted element-wise multiplication.
  Adapted from: https://arxiv.org/pdf/1709.01507.pdf
</div>

```python3
se = co.Residual(
    co.Sequential(
        OrderedDict([
            ("pool", co.AdaptiveAvgPool3d((1, 1, 1), kernel_size=7)),
            ("down", co.Conv3d(256, 16, kernel_size=1)),
            ("act1", nn.ReLU()),
            ("up", co.Conv3d(16, 256, kernel_size=1)),
            ("act2", nn.Sigmoid()),
        ])
    ),
    aggregation_fn="mul",
)
```


## Compatibility
The library modules are built to integrate seamlessly with other PyTorch projects.
Specifically, extra care was taken to ensure out-of-the-box compatibility with:
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)
- [ride](https://github.com/LukasHedegaard/ride)


## Citations
This library
```bibtex
@article{hedegaard2021colib,
  title={Continual Inference Library},
  author={Lukas Hedegaard},
  journal={GitHub. Note: https://github.com/LukasHedegaard/continual-inference},
  year={2021}
}
```

[Continual 3D CNNs](https://github.com/LukasHedegaard/co3d) [![Paper](http://img.shields.io/badge/paper-arxiv.2106.00050-B31B1B.svg)](https://arxiv.org/abs/2106.00050)
```bibtex
@article{hedegaard2021co3d,
  title={Continual 3D Convolutional Neural Networks for Real-time Processing of Videos},
  author={Lukas Hedegaard and Alexandros Iosifidis},
  journal={preprint, arXiv:2106.00050},
  year={2021}
}
```

<!-- [Continual Skeletons](https://github.com/LukasHedegaard/continual-skeletons)
```bibtex
@article{hedegaard2021coskelleton,
  title={Continual Skeletons for Efficient Online Activity Recognition},
  author={Lukas Hedegaard and Negar Heidari and Alexandros Iosifidis},
  journal={TBD},
  year={2021}
}
``` -->
