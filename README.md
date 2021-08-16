# Continual Inference
Building blocks for Continual Inference Networks in PyTorch

<div align="left">
  <a href="https://pypi.org/project/continual-inference/">
    <img src="https://img.shields.io/pypi/pyversions/continual-inference" height="20" >
  </a>
  <a href="https://pypi.org/project/continual-inference/">
    <img src="https://github.com/LukasHedegaard/continual-inference/workflows/Python%20package/badge.svg" height="20" >
  </a>
  <a href="https://codecov.io/gh/LukasHedegaard/continual-inference">
    <img src="https://codecov.io/gh/LukasHedegaard/continual-inference/branch/main/graph/badge.svg?token=XW1UQZSEOG"/>
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
  <br><br>
</div>

## Install 
```bash
pip install continual-inference
```

## Usage
```python3
import torch
from torch import nn
import continual as co
                                                           # B, C, T, H, W
example = torch.normal(mean=torch.zeros(5 * 3 * 3)).reshape((1, 1, 5, 3, 3))

  # Acts as a drop-in replacement for torch.nn modules ✅
  co_conv = co.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3))
  nn_conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3))
  co_conv.load_state_dict(nn_conv.state_dict()) # ensure identical weights

  co_output = co_conv(example)  # Same exact computation
  nn_output = nn_conv(example)  # Same exact computation
  assert torch.equal(co_output, nn_output)

  # But can also perform online inference efficiently 🚀
  firsts = co_conv.forward_steps(example[:, :, :4])
  last = co_conv.forward_step(example[:, :, 4])

  assert torch.allclose(nn_output[:, :, : co_conv.delay], firsts)
  assert torch.allclose(nn_output[:, :, co_conv.delay], last)
```

## Continual Inference Networks (CINs)
Continual Inference Networks are a type of neural network, which operate on a continual input stream of data and infer a new prediction for each new time-step.

All networks and network-modules, that do not utilise temporal information can be used for an Online Inference Network (e.g. `nn.Conv1d` and `nn.Conv2d` on spatial data such as an image). 
Moreover, recurrent modules (e.g. `LSTM` and `GRU`), that summarize past events in an internal state are also useable in CINs.

__CIN__:
```
  O       O       O      (output)
  ↑       ↑       ↑       
LSTM    LSTM    LSTM     (temporal LSTM)
  ↑       ↑       ↑    
Conv2D  Conv2D  Conv2D   (spatial 2D conv)
  ↑       ↑       ↑    
  I       I       I      (input frame)
```

However, modules that operate on temporal data with the assumption that the more temporal context is available than the current frame (e.g. the spatio-temporal `nn.Conv3d` used by many SotA video recognition models) cannot be directly applied.

__Not CIN__:
```
          Θ              (output)   
          ↑              
        Conv3D           (spatio-temporal 3D conv)
          ↑
  -----------------      (concatenate frames to clip)
  ↑       ↑       ↑    
  I       I       I      (input frame)  
```

Sometimes, though, the computations in such modules, can be cleverly restructured to work for online inference as well! 

__CIN__:
```
   O         O         Θ      (output)
   ↑         ↑         ↑    
ConvCo3D  ConvCo3D  ConvCo3D  (continual spatio-temporal 3D conv)
   ↑         ↑         ↑    
   I         I         I      (input frame)
```
Here, the `ϴ` output of the `Conv3D` and `ConvCo3D` are identical! ✨

## Modules
This repository contains online inference-friendly versions of common network building blocks, inlcuding:

<!-- TODO: Replace with link to docs once they are set up -->
- (Temporal) convolutions:
    - `co.Conv1d`
    - `co.Conv2d`
    - `co.Conv3d`

- (Temporal) batch normalisation:
    - `co.BatchNorm2d`

- (Temporal) pooling:
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

- Other
    - `co.Sequential` - sequential wrapper for modules
    - `co.Parallel` - parallel wrapper for modules
    - `co.Residual` - residual wrapper for modules
    - `co.Delay` - pure delay module
    <!-- - `co.Residual` - residual connection, which automatically adds delay if needed -->
    - `co.unsqueezed` - functional wrapper for non-continual modules
    - `co.continual` - conversion function from non-continual modules to continual moduls

### Continual Convolutions
Continual Convolutions can lead to major improvements in computational efficiency when online / frame-by-frame predictions are required.

Below, principle sketches comparing regular and continual convolutions are shown:

<div align="center">
<img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/continual-convolution.png" width="500">
  <br>
  Regular Convolution. 
	A regular temporal convolutional layer leads to redundant computations during online processing of video clips, as illustrated by the repeated convolution of inputs (green b,c,d) with a kernel (blue α,β) in the temporal dimen- sion. Moreover, prior inputs (b,c,d) must be stored be- tween time-steps for online processing tasks.
  <br><br>  
  <img src="https://raw.githubusercontent.com/LukasHedegaard/continual-inference/main/figures/regular-convolution.png" width="500">
  <br>
  Continual Convolution. 
	An input (green d or e) is convolved with a kernel (blue α, β). The intermediary feature-maps corresponding to all but the last temporal position are stored, while the last feature map and prior memory are summed to produce the resulting output. For a continual stream of inputs, Continual Convolutions produce identical outputs to regular convolutions.
  <br><br>
</div>


For more information, we refer to the [seminal paper on Continual Convolutions](https://arxiv.org/abs/2106.00050).

## Forward modes
The library components feature three distinct forward modes, which are handy for different situations.

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

## Compatibility
The library modules are built to integrate seamlessly with other PyTorch projects.
Specifically, extra care was taken to ensure out-of-the-box compatibility with:
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)
- [ride](https://github.com/LukasHedegaard/ride)


## Projects
For full-fledged examples of complex Continual Inference Networks, see:

- [Continual 3D](https://github.com/LukasHedegaard/co3d)
<!-- - [Continual Skeletons](https://github.com/LukasHedegaard/continual-skeletons) -->


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