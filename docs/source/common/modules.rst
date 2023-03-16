The the modules for Continual Inference Networks are listed below.
They are designed to be drop-in replacements for the `torch.nn` modules of the same name.
Methods of the same name have identical interfaces and execute identical code. 
The modules are extended with the `forward_step` and `forward_steps` functions alongside common properties as found in :class:`continual.CoModule`.


.. role:: hidden
    :class: hidden-section


.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. automodule:: continual


Containers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    CoModule
    Sequential
    Broadcast
    Parallel
    ParallelDispatch
    Reduce
    BroadcastReduce
    Residual
    Conditional
    

Convolution Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Conv1d
    Conv2d
    Conv3d


Pooling Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AvgPool1d
    AvgPool2d
    AvgPool3d
    MaxPool1d
    MaxPool2d
    MaxPool3d
    AdaptiveAvgPool2d
    AdaptiveAvgPool3d
    AdaptiveMaxPool2d
    AdaptiveMaxPool3d

Recurrent Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    RNN
    LSTM
    GRU

Transformer Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TransformerEncoder
    TransformerEncoderLayerFactory
    SingleOutputTransformerEncoderLayer
    RetroactiveTransformerEncoderLayer
    RetroactiveMultiheadAttention
    SingleOutputMultiheadAttention
    RecyclingPositionalEncoding

Linear Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
    Identity
    Add
    Multiply


Utilities
---------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Lambda
    Delay
    Skip
    Reshape
    Constant
    Zero
    One


Converters
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    continual
    forward_stepping
    call_mode