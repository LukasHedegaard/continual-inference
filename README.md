<img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/logo/logo_name.svg" style="width: 30vw; min-width: 400px;">

__A Python library for Continual Inference Networks in PyTorch__

[Quick-start](#-quick-start) • 
[Docs](https://continual-inference.readthedocs.io) • 
[Paper](https://arxiv.org/abs/2204.03418) • 
[Examples](#-network-composition) • 
[Modules](#-module-library) • 
[Model Zoo](#-model-zoo) • 
[Contribute]() • 
[License](LICENSE)

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
  <!-- <a href="https://arxiv.org/abs/2204.03418">
    <img src="http://img.shields.io/badge/paper-arxiv.2204.03418-B31B1B.svg" height="20" >
  </a> -->
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
  <a href="https://www.codefactor.io/repository/github/lukashedegaard/continual-inference/overview/main">
    <img src="https://www.codefactor.io/repository/github/lukashedegaard/continual-inference/badge/main" alt="CodeFactor" height="20" />
  </a>
  <sup>*</sup>
</div>

###### \*We match PyTorch interfaces exactly. This reduces the codefactor to "A-" due to method arguments named "input".

## Continual Inference Networks ensure efficient stream processing
Many of our favorite Deep Neural Network architectures (e.g., [CNNs](https://arxiv.org/abs/2106.00050) and [Transformers](https://arxiv.org/abs/2201.06268)) were built with offline-processing for offline processing. Rather than processing inputs one sequence element at a time, they require the whole (spatio-)temporal sequence to be passed as a single input.
Yet, **many important real-life applications need online predictions on a continual input stream**. 
While CNNs and Transformers can be applied by re-assembling and passing sequences within a sliding window, this is _inefficient_ due to the redundant intermediary computations from overlapping clips.

**Continual Inference Networks** (CINs) ensure efficient stream processing via an alternative computational ordering, with ~_L_ ×  fewer FLOPs per prediction compared to sliding window-based inference with non-CINs where _L_ is the corresponding sequence length of a non-CIN network. For details on their inner workings, check out the videos below or the corresponding papers [[1](https://arxiv.org/abs/2106.00050), [2](https://arxiv.org/abs/2201.06268)].


<div align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=Jm2A7dVEaF4" target="_blank">
     <img src="http://img.youtube.com/vi/Jm2A7dVEaF4/hqdefault.jpg" alt="1 minute overview" width="240" height="180" border="0" />
  </a>
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=gy802Tlp-eQ" target="_blank">
     <img src="http://img.youtube.com/vi/gy802Tlp-eQ/hqdefault.jpg" alt="1 minute overview" width="240" height="180" border="0" />
  </a>
</div>

## News
- 2022-12-02: ONNX compatibility for all modules is available from v1.0.0. See [test_onnx.py](tests/continual/test_onnx.py) for examples.


## Quick-start

### Install 
```bash
pip install continual-inference
```



### Example
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

See the [network composition](#-composition) and [model zoo](#model-zoo) sections for additional examples.

## Library principles

### Forward modes
The library components feature three distinct forward modes, which are handy for different situations, namely `forward`, `forward_step`, and `forward_steps`:

#### `forward`
Performs a full forward computation exactly as the regular layer would.
This method is handy for efficient training on clip-based data.

```
         O            (O: output)
         ↑ 
         N            (N: network module)
         ↑ 
 -----------------    (-: aggregation)
 P   I   I   I   P    (I: input frame, P: padding)
```


#### `forward_step`
Performs a forward computation for a single frame and continual states are updated accordingly. This is the mode to use for continual inference.

```
O+S O+S O+S O+S   (O: output, S: updated internal state)
 ↑   ↑   ↑   ↑ 
 N   N   N   N    (N: network module)
 ↑   ↑   ↑   ↑ 
 I   I   I   I    (I: input frame)
```

#### `forward_steps`
Performs a layer-wise forward computation using the continual module.
The computation is performed frame-by-frame and continual states are updated accordingly.
The output-input mapping the exact same as that of a regular module.
This mode is handy for initializing the network on a whole clip (multiple frames) before the `forward` is used for continual inference. 
```
         O            (O: output)
         ↑ 
 -----------------    (-: aggregation)
 O  O+S O+S O+S  O    (O: output, S: updated internal state)
 ↑   ↑   ↑   ↑   ↑
 N   N   N   N   N    (N: network module)
 ↑   ↑   ↑   ↑   ↑
 P   I   I   I   P    (I: input frame, P: padding)
```



### Composition

Continual Inference Networks require strict handling of internal data delays to guarantee correspondence between [forward modes](#-forward-modes). While it is possible to composed neural networks with an imperative programming style, correct handling of delays is cumbersome and time-consuming. Instead, we provide a rich interface of modules for powerful module composition using a _functional style_, which handles delays automatically. Think extensions or `torch.nn.Sequential`.

_Composition examples_ can be expanded below:

<details>
<summary><b>Residual module</b></summary>

Short-hand:
```python3
residual = co.Residual(co.Conv3d(32, 32, kernel_size=3, padding=1))
```

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

</details>

<details>
<summary><b>3D MobileNetV2 Inverted residual block</b></summary>

Continual 3D version of the [MobileNetV2 Inverted residual block](https://arxiv.org/pdf/1801.04381.pdf).

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

</details>

<details>
<summary><b>3D Squeeze-and-Excitation module</b></summary>

Continual 3D version of the [Squeeze-and-Excitation module](https://arxiv.org/pdf/1709.01507.pdf)

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

</details>

<details>
<summary><b>3D Inception module</b></summary>

Continual 3D version of the [Inception module](https://arxiv.org/pdf/1409.4842v1.pdf):
<div align="center">
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/examples/inception_block.png" style="width: 25vw; min-width: 350px;">
  <br>
  Inception module. Source: https://arxiv.org/pdf/1409.4842v1.pdf
   
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
</details>
</br>

## Module library
_Continual Inference_ features a rich library of modules for defining Continual Inference Networks. Specific care was taken to create CIN versions of the PyTorch modules found in [_torch.nn_](https://pytorch.org/docs/stable/nn.html):

<details>
<summary><b>Convolutions</b></summary>

- [`co.Conv1d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/conv/index.html#continual.conv.Conv1d)
- [`co.Conv2d`]((https://continual-inference.readthedocs.io/en/latest/autoapi/continual/conv/index.html#continual.conv.Conv2d))
- [`co.Conv3d`]((https://continual-inference.readthedocs.io/en/latest/autoapi/continual/conv/index.html#continual.conv.Conv3d))

</details>

<details>
<summary><b>Pooling</b></summary>

  - [`co.AvgPool1d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AvgPool1d)
  - [`co.AvgPool2d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AvgPool2d)
  - [`co.AvgPool3d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AvgPool3d)
  - [`co.MaxPool1d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.MaxPool1d)
  - [`co.MaxPool2d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.MaxPool2d)
  - [`co.MaxPool3d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.MaxPool3d)
  - [`co.AdaptiveAvgPool2d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AdaptiveAvgPool2d)
  - [`co.AdaptiveAvgPool3d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AdaptiveAvgPool3d)
  - [`co.AdaptiveMaxPool2d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AdaptiveMaxPool2d)
  - [`co.AdaptiveMaxPool3d`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/pooling/index.html#continual.pooling.AdaptiveMaxPool3d)

</details>

<details>
<summary><b>Linear</b></summary>

  - [`co.Linear`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/linear/index.html#continual.linear.Linear)

</details>

<details>
<summary><b>Recurrent</b></summary>

  - [`co.RNN`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/rnn/index.html#continual.rnn.RNN)
  - [`co.LSTM`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/rnn/index.html#continual.rnn.LSTM)
  - [`co.GRU`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/rnn/index.html#continual.rnn.GRU)

</details>

<details>
<summary><b>Transformers</b></summary>

  - [`co.TransformerEncoder`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/transformer/index.html#continual.transformer.TransformerEncoder)
  - [`co.TransformerEncoderLayerFactory`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/transformer/index.html#continual.transformer.TransformerEncoderLayerFactory): Factory function corresponding to `nn.TransformerEncoderLayer`.
  - [`co.SingleOutputTransformerEncoderLayer`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/transformer/index.html#continual.transformer.SingleOutputTransformerEncoderLayer): SingleOutputMHA version of `nn.TransformerEncoderLayer`.
  - [`co.RetroactiveTransformerEncoderLayer`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/transformer/index.html#continual.transformer.RetroactiveTransformerEncoderLayer): RetroactiveMHA version of `nn.TransformerEncoderLayer`.
  - [`co.RetroactiveMultiheadAttention`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/multihead_attention/retroactive_mha/index.html#continual.multihead_attention.retroactive_mha.RetroactiveMultiheadAttention): Retroactive version of `nn.MultiheadAttention`.
  - [`co.SingleOutputMultiheadAttention`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/multihead_attention/single_output_mha/index.html#continual.multihead_attention.single_output_mha.SingleOutputMultiheadAttention): Single-output version of `nn.MultiheadAttention`.
  - [`co.RecyclingPositionalEncoding`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/positional_encoding/index.html#continual.positional_encoding.RecyclingPositionalEncoding): Positional Encoding used for Continual Transformers.

</details>

</br>

Modules for composing and converting networks. Both _composition_ and _utility_ modules can be used for regular definition of PyTorch modules as well.

<details>
<summary><b>Composition modules</b></summary>

  - [`co.Sequential`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.Sequential): Sequential wrapper for modules. This module automatically performs conversions of torch.nn modules, which are safe during continual inference. These include all batch normalization and activation function. 
  - [`co.Residual`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.Residual): Residual wrapper for modules.
  - [`co.BroadcastReduce`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.BroadcastReduce): BroadcastReduce wrapper for modules.
  - [`co.Broadcast`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.Broadcast): Broadcast one stream to multiple.
  - [`co.Parallel`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.Parallel): Parallel wrapper for modules. Like `co.Sequential`, this module performs automatic conversion of torch.nn modules.
  - [`co.ParallelDispatch`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.ParallelDispatch): Parallel dispatch of many input streams to many output streams.
  - [`co.Reduce`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.Reduce): Reduce multiple input streams to one.
  - [`co.Conditional`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/container/index.html#continual.container.Conditional): Conditionally checks whether to invoke a module at runtime.

</details>

<details>
<summary><b>Utility modules</b></summary>

  - [`co.Delay`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/delay/index.html#continual.delay.Delay): Pure delay module (e.g. needed in residuals).
  - [`co.Reshape`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/shape/index.html#continual.shape.Reshape): Reshape non-temporal dimensions.
  - [`co.Lambda`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.Lambda): Lambda module which wraps any function.
  - [`co.Add`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.Add): Adds a constant value.
  - [`co.Multiply`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.Multiply): Multiplies with a constant factor.
  - [`co.Unity`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.Unity): Maps input to output without modification.
  - [`co.Constant`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.Constant): Maps input to and output with constant value.
  - [`co.Zero`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.Zero): Maps input to output of zeros.
  - [`co.One`](https://continual-inference.readthedocs.io/en/latest/autoapi/continual/closure/index.html#continual.closure.One): Maps input to output of ones.

</details>

<details>
<summary><b>Converters</b></summary>

  - [`co.continual`](): conversion function from `torch.nn` modules to `co` modules.
  - [`co.forward_stepping`](): functional wrapper, which enhances temporally local `torch.nn` modules with the forward_stepping functions.

</details>

</br>

We support drop-in interoperability with with the following _torch.nn_ modules:

<details>
<summary><b>Activation</b></summary>

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

</details>

<details>
<summary><b>Normalization</b></summary>

  - `nn.BatchNorm1d`
  - `nn.BatchNorm2d`
  - `nn.BatchNorm3d`
  - `nn.LayerNorm`

</details>

<details>
<summary><b>Dropout</b></summary>

  - `nn.Dropout`
  - `nn.Dropout2d`
  - `nn.Dropout3d`
  - `nn.AlphaDropout`
  - `nn.FeatureAlphaDropout`

</details>
</br>


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
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)
- [ride](https://github.com/LukasHedegaard/ride)
- [onnx](https://github.com/onnx/onnx)
<!-- - [onnxruntime](https://github.com/microsoft/onnxruntime) -->


## Citation
<div align="left">
<a href="https://arxiv.org/abs/2204.03418">
  <img src="http://img.shields.io/badge/paper-arxiv.2204.03418-B31B1B.svg" height="20" >
</a>
</div>

```bibtex
@inproceedings{hedegaard2022colib,
  title={Continual Inference: A Library for Efficient Online Inference with Deep Neural Networks in PyTorch},
  author={Lukas Hedegaard and Alexandros Iosifidis},
  booktitle={European Conference on Computer Vision Workshops (ECCVW)},
  year={2022}
}
```


## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR).
