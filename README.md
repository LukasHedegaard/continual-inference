<img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/logo/logo_name.svg" style="width: 30vw; min-width: 400px;">

__PyTorch building blocks for Continual Inference Networks__

<div align="left">
  <a href="https://pypi.org/project/continual-inference/">
    <img src="https://img.shields.io/pypi/pyversions/continual-inference" height="20" >
  </a>
  <a href="https://badge.fury.io/py/continual-inference">
    <img src="https://badge.fury.io/py/continual-inference.svg" height="20" >
  </a>
  <a href="https://continual-inference.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/continual-inference/badge/?version=latest" alt="Documentation Status" height="20"/>
  </a>
  <a href="https://pepy.tech/project/continual-inference">
    <img src="https://pepy.tech/badge/continual-inference" height="20">
  </a>
  <a href="https://codecov.io/gh/LukasHedegaard/continual-inference">
    <img src="https://codecov.io/gh/LukasHedegaard/continual-inference/branch/main/graph/badge.svg?token=XW1UQZSEOG" height="20"/>
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://arxiv.org/abs/2204.03418">
    <img src="http://img.shields.io/badge/paper-arxiv.2204.03418-B31B1B.svg" height="20" >
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
  <a href="https://www.codefactor.io/repository/github/lukashedegaard/continual-inference/overview/main">
    <img src="https://www.codefactor.io/repository/github/lukashedegaard/continual-inference/badge/main" alt="CodeFactor" height="20" />
  </a>
  <sup>*</sup>
</div>

###### \*We match PyTorch interfaces exacly. This reduces the codefactor to "A-" due to method arguments named "input".

## Install 
```bash
pip install continual-inference
```

## News
- 2022-12-02: ONNX compatibility for all modules is available from v1.0.0. See [test_onnx.py](tests/continual/test_onnx.py) for examples.
- 2022-08-18: The library paper ["Continual Inference: A Library for Efficient Online Inference with Deep Neural Networks in PyTorch"](https://arxiv.org/abs/2204.03418) was accepted at the ECCV 2022 workhop on Computational Aspects of Deep Learning. 🎉

## A motivating example
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

# Temporal properties
assert conv.receptive_field == 3
assert conv.delay == 2
```

For more examples, see the [Advanced Module Examples](#advanced-module-examples) and [Model Zoo](#model-zoo).

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
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/continual/regular-convolution.png" style="width: 25vw; min-width: 350px;">
  <br>
  (1) <br> 
  Regular Convolution. 
	A regular temporal convolutional layer leads to redundant computations during online processing of video clips, as illustrated by the repeated convolution of inputs (green b,c,d) with a kernel (blue α,β) in the temporal dimension. Moreover, prior inputs (b,c,d) must be stored between time-steps for online processing tasks.
  <br><br>
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/continual/continual-convolution.png" style="width: 25vw; min-width: 350px;">
  <br>
  (2) <br>
  Continual Convolution. 
	An input (green d or e) is convolved with a kernel (blue α, β). The intermediary feature-maps corresponding to all but the last temporal position are stored, while the last feature map and prior memory are summed to produce the resulting output. For a continual stream of inputs, Continual Convolutions produce identical outputs to regular convolutions.
  <br><br>  
</div>

Comparing Figures (1) and (2), we see that Continual Convolutions get rid of computational redundancies.
This can speed up online inference greatly - for example, a Continual X3D model for Human Activity Recognition has __10× less Floating Point Operations per prediction__ than the vanilla X3D models 🚀. 

> 💡  The longer the length of the temporal sequence, the larger the savings.

For more information, we refer to the [paper describing this library](https://arxiv.org/abs/2204.03418).


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
Below is a list of the modules and utilities included in the library:

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

- Linear:
    - `co.Linear`

- Recurrent:
    - `co.RNN`
    - `co.LSTM`
    - `co.GRU`

- Transformers:
    - `co.TransformerEncoder`
    - `co.TransformerEncoderLayerFactory` - Factory function corresponding to `nn.TransformerEncoderLayer`.
    - `co.SingleOutputTransformerEncoderLayer` - SingleOutputMHA version of `nn.TransformerEncoderLayer`.
    - `co.RetroactiveTransformerEncoderLayer` - RetroactiveMHA version of `nn.TransformerEncoderLayer`.
    - `co.RetroactiveMultiheadAttention` - Retroactive version of `nn.MultiheadAttention`.
    - `co.SingleOutputMultiheadAttention` - Single-output version of `nn.MultiheadAttention`.
    - `co.RecyclingPositionalEncoding` - Positional Encoding used for Continual Transformers.

- Containers
    - `co.Sequential` - Sequential wrapper for modules. This module automatically performs conversions of torch.nn modules, which are safe during continual inference. These include all batch normalisation and activation function. 
    - `co.Broadcast` - Broadcast one stream to multiple.
    - `co.Parallel` - Parallel wrapper for modules. Like `co.Sequential`, this module performs automatic conersion of torch.nn modules.
    - `co.ParallelDispatch` - Parallel dispatch of many input streams to many output streams.
    - `co.Reduce` - Reduce multiple input streams to one.
    - `co.Residual` - Residual wrapper for modules.
    - `co.BroadcastReduce` - BroadcastReduce wrapper for modules.
    - `co.Conditional` - Conditionally checks whether to invoke a module at runtime.

- Other
    - `co.Delay` - Pure delay module (e.g. needed in residuals).
    - `co.Reshape` - Reshape non-temporal dimensions.
    - `co.Lambda` - Lambda module which wraps any function.
    - `co.Add` - Adds a constant value.
    - `co.Multiply` - Multiplies with a constant factor.
    - `co.Unity` - Maps input to output without modification.
    - `co.Constant` - Maps input to and output with constant value.
    - `co.Zero` - Maps input to output of zeros.
    - `co.One` - Maps input to output of ones.

- Converters
    - `co.continual` - conversion function from `torch.nn` modules to `co` modules.
    - `co.forward_stepping` - functional wrapper, which enhances temporally local `torch.nn` modules with the forward_stepping functions.

In addition, we support interoperability with a wide range of modules from `torch.nn`:

- Activation
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

- Normalisation
    - `nn.BatchNorm1d`
    - `nn.BatchNorm2d`
    - `nn.BatchNorm3d`
    - `nn.LayerNorm`

- Dropout
    - `nn.Dropout`
    - `nn.Dropout2d`
    - `nn.Dropout3d`
    - `nn.AlphaDropout`
    - `nn.FeatureAlphaDropout`

## Advanced module examples

### Residual module
Explicit:
```python3
residual = co.Sequential(
    co.Broadcast(2),
    co.Parallel(
        co.Conv3d(32, 32, kernel_size=3, padding=1),
        co.Delay(2),
    ),
    co.Reduce("sum"),
)
```

Short-hand:
```python3
residual = co.Residual(co.Conv3d(32, 32, kernel_size=3, padding=1))
```

### Continual 3D [MBConv](https://arxiv.org/pdf/1801.04381.pdf)

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/mb_conv.png" style="width: 15vw; min-width: 200px;">
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


### Continual 3D [Squeeze-and-Excitation module](https://arxiv.org/pdf/1709.01507.pdf)

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/se_block.png" style="width: 15vw; min-width: 200px;">
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
    reduce="mul",
)
```


### Continual 3D [Inception module](https://arxiv.org/pdf/1409.4842v1.pdf)

<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/inception_block.png" style="width: 25vw; min-width: 350px;">
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

inception_module = co.BroadcastReduce(
    co.Conv3d(192, 64, kernel_size=1),
    co.Sequential(
        norm_relu(co.Conv3d(192, 96, kernel_size=1), 96),
        norm_relu(co.Conv3d(96, 128, kernel_size=3, padding=1), 128),
    ),
    co.Sequential(
        norm_relu(co.Conv3d(192, 16, kernel_size=1), 16),
        norm_relu(co.Conv3d(16, 32, kernel_size=5, padding=2), 32),
    ),
    co.Sequential(
        co.MaxPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
        norm_relu(co.Conv3d(192, 32, kernel_size=1), 32),
    ),
    reduce="concat",
)
```

## Model Zoo
### Continual 3D CNNs
- [_Co_ X3D](https://github.com/LukasHedegaard/co3d/tree/main/models/cox3d)
- [_Co_ Slow](https://github.com/LukasHedegaard/co3d/tree/main/models/coslow)
- [_Co_ I3D](https://github.com/LukasHedegaard/co3d/tree/main/models/coi3d)

### Continual ST-GCNs
- [_Co_ STGCN](https://github.com/LukasHedegaard/continual-skeletons/blob/main/models/cost_gcn_mod/cost_gcn_mod.py)
- [_Co_ AGCN](https://github.com/LukasHedegaard/continual-skeletons/blob/main/models/coa_gcn_mod/coa_gcn_mod.py)
- [_Co_ STr](https://github.com/LukasHedegaard/continual-skeletons/blob/main/models/cos_tr_mod/cos_tr_mod.py)

### Continual Transformers
- [_Continual_ One-block Transformer Encoder](https://github.com/LukasHedegaard/continual-inference/blob/9895344f50a93ebb5cf5c4f26ecfdf27b6a3fe75/tests/continual/test_transformer.py#L8)
- [_Continual_ Two-block Transformer Encoder](https://github.com/LukasHedegaard/continual-inference/blob/9895344f50a93ebb5cf5c4f26ecfdf27b6a3fe75/tests/continual/test_transformer.py#L59)




## Compatibility
The library modules are built to integrate seamlessly with other PyTorch projects.
Specifically, extra care was taken to ensure out-of-the-box compatibility with:
- [onnx](https://github.com/onnx/onnx)
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)
- [ride](https://github.com/LukasHedegaard/ride)


## Citation
If you use this library or the continual modules, please consider citing:

### This library
<div align="left">
<a href="https://arxiv.org/abs/2204.03418">
  <img src="http://img.shields.io/badge/paper-arxiv.2204.03418-B31B1B.svg" height="20" >
</a>
</div>

```bibtex
@article{hedegaard2022colib,
  title={Continual Inference: A Library for Efficient Online Inference with Deep Neural Networks in PyTorch},
  author={Lukas Hedegaard and Alexandros Iosifidis},
  journal={preprint, arXiv:2204.03418},
  year={2022}
}
```

### Continual Convolutions
<div align="left">
<a href="https://arxiv.org/abs/2106.00050">
  <img src="http://img.shields.io/badge/paper-arxiv.2106.00050-B31B1B.svg" height="20" >
</a>
</div>

```bibtex
@inproceedings{hedegaard2022co3d,
    title={Continual 3D Convolutional Neural Networks for Real-time Processing of Videos},
    author={Lukas Hedegaard and Alexandros Iosifidis},
    pages={1--18},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022},
}
```

<div align="left">
<a href="https://arxiv.org/abs/2203.11009">
  <img src="http://img.shields.io/badge/paper-arxiv.2203.11009-B31B1B.svg" height="20" >
</a>
</div>

```bibtex
@article{hedegaard2022online,
  title={Online Skeleton-based Action Recognition with Continual Spatio-Temporal Graph Convolutional Networks},
  author={Lukas Hedegaard and Negar Heidari and Alexandros Iosifidis},
  journal={preprint, arXiv: 2203.11009}, 
  year={2022}
}
```

### Continual Transformers
<div align="left">
<a href="https://arxiv.org/abs/2201.06268">
  <img src="http://img.shields.io/badge/paper-arxiv.2201.06268-B31B1B.svg" height="20" >
</a>
</div>

```bibtex
@article{hedegaard2022cotrans,
  title={Continual Transformers: Redundancy-Free Attention for Online Inference},
  author={Lukas Hedegaard and Alexandros Iosifidis},
  journal={preprint, arXiv:2201.06268},
  year={2022}
}
```

## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR).
