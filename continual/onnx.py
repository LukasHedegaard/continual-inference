import io
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch._C._onnx as _C_onnx

from continual.module import CoModule
from continual.utils import flatten

__all__ = ["export", "OnnxWrapper"]


def export(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    export_params: bool = True,
    verbose: bool = False,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    # dynamic_axes: Optional[  # Handled automatically
    #     Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    # ] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    # export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]] = False,
):
    r"""Exports a model into ONNX format.
    If ``model`` is not a :class:`torch.jit.ScriptModule` nor a
    :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.
    Args:
        model (:class:`torch.nn.Module`, :class:`torch.jit.ScriptModule` or :class:`torch.jit.ScriptFunction`):
            the model to be exported.
        args (tuple or torch.Tensor):
            args can be structured either as:
            1. ONLY A TUPLE OF ARGUMENTS::
                args = (x, y, z)
            The tuple should contain model inputs such that ``model(*args)`` is a valid
            invocation of the model. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.
            2. A TENSOR::
                args = torch.Tensor([1])
            This is equivalent to a 1-ary tuple of that Tensor.
            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS::
                args = (
                    x,
                    {
                        "y": input_y,
                        "z": input_z
                    }
                )
            All but the last element of the tuple will be passed as non-keyword arguments,
            and named arguments will be set from the last element. If a named argument is
            not present in the dictionary, it is assigned the default value, or None if a
            default value is not provided.
            .. note::
                If a dictionary is the last element of the args tuple, it will be
                interpreted as containing named arguments. In order to pass a dict as the
                last non-keyword arg, provide an empty dict as the last element of the args
                tuple. For example, instead of::
                    torch.onnx.export(
                        model,
                        (
                            x,
                            # WRONG: will be interpreted as named arguments
                            {y: z}
                        ),
                        "test.onnx.pb"
                    )
                Write::
                    torch.onnx.export(
                        model,
                        (
                            x,
                            {y: z},
                            {}
                        ),
                        "test.onnx.pb"
                    )
        f: a file-like object (such that ``f.fileno()`` returns a file descriptor)
            or a string containing a file name.  A binary protocol buffer will be written
            to this file.
        export_params (bool, default True): if True, all parameters will
            be exported. Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, with the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if True, prints a description of the
            model being exported to stdout. In addition, the final ONNX graph will include the
            field ``doc_string``` from the exported model which mentions the source code locations
            for ``model``. If True, ONNX exporter logging will be turned on.
        training (enum, default TrainingMode.EVAL):
            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode if model.training is
                False and in training mode if model.training is True.
            * ``TrainingMode.TRAINING``: export the model in training mode. Disables optimizations
                which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order.
        operator_export_type (enum, default OperatorExportTypes.ONNX):
            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
                (in the default opset domain).
            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
                to standard ONNX ops in the default opset domain. If unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting the op into a custom opset domain without conversion. Applies
                to `custom ops <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
                as well as ATen ops. For the exported model to be usable, the runtime must support
                these non-standard ops.
            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript namespace "aten")
                are exported as ATen ops (in opset domain "org.pytorch.aten").
                `ATen <https://pytorch.org/cppdocs/#aten>`_ is PyTorch's built-in tensor library, so
                this instructs the runtime to use PyTorch's implementation of these ops.
                .. warning::
                    Models exported this way are probably runnable only by Caffe2.
                    This may be useful if the numeric differences in implementations of operators are
                    causing large differences in behavior between PyTorch and Caffe2 (which is more
                    common on untrained models).
            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each ATen op
                (in the TorchScript namespace "aten") as a regular ONNX op. If we are unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting an ATen op. See documentation on OperatorExportTypes.ONNX_ATEN for
                context.
                For example::
                    graph(%0 : Float):
                    %3 : int = prim::Constant[value=0]()
                    # conversion unsupported
                    %4 : Float = aten::triu(%0, %3)
                    # conversion supported
                    %5 : Float = aten::mul(%4, %0)
                    return (%5)
                Assuming ``aten::triu`` is not supported in ONNX, this will be exported as::
                    graph(%0 : Float):
                    %1 : Long() = onnx::Constant[value={0}]()
                    # not converted
                    %2 : Float = aten::ATen[operator="triu"](%0, %1)
                    # converted
                    %3 : Float = onnx::Mul(%2, %0)
                    return (%3)
                If PyTorch was built with Caffe2 (i.e. with ``BUILD_CAFFE2=1``), then
                Caffe2-specific behavior will be enabled, including special support
                for ops are produced by the modules described in
                `Quantization <https://pytorch.org/docs/stable/quantization.html>`_.
                .. warning::
                    Models exported this way are probably runnable only by Caffe2.
        opset_version (int, default 14): The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. Must be >= 7 and <= 16.
        do_constant_folding (bool, default True): Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes.
        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.
            If ``opset_version < 9``, initializers MUST be part of graph
            inputs and this argument will be ignored and the behavior will be
            equivalent to setting this argument to True.
            If None, then the behavior is chosen automatically as follows:
            * If ``operator_export_type=OperatorExportTypes.ONNX``, the behavior is equivalent
                to setting this argument to False.
            * Else, the behavior is equivalent to setting this argument to True.
        custom_opsets (dict[str, int], default empty dict): A dict with schema:
            * KEY (str): opset domain name
            * VALUE (int): opset version
            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument.
        export_modules_as_functions (bool or set of type of nn.Module, default False): Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes.
            1. Annotated attributes: class variables that have type annotations via
            `PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_
            will be exported as attributes.
            Annotated attributes are not used inside the subgraph of ONNX local function because
            they are not created by PyTorch JIT tracing, but they may be used by consumers
            to determine whether or not to replace the function with a particular fused kernel.
            2. Inferred attributes: variables that are used by operators inside the module. Attribute names
            will have prefix "inferred::". This is to differentiate from predefined attributes retrieved from
            python module annotations. Inferred attributes are used inside the subgraph of ONNX local function.
            * ``False`` (default): export ``nn.Module`` forward calls as fine grained nodes.
            * ``True``: export all ``nn.Module`` forward calls as local function nodes.
            * Set of type of nn.Module: export ``nn.Module`` forward calls as local function nodes,
                only if the type of the ``nn.Module`` is found in the set.
    Raises:
        :class:`torch.onnx.errors.CheckerError`: If the ONNX checker detects an invalid ONNX graph.
        :class:`torch.onnx.errors.UnsupportedOperatorError`: If the ONNX graph cannot be exported because it
            uses an operator that is not supported by the exporter.
        :class:`torch.onnx.errors.OnnxExporterError`: Other errors that can occur during export.
            All errors are subclasses of :class:`errors.OnnxExporterError`.
    """
    assert isinstance(
        model, CoModule
    ), f"The passed model of type {type(model)} should be a CoModule"
    omodel = OnnxWrapper(model)
    torch.onnx.export(
        model=omodel,
        args=args,
        f=f,
        export_params=export_params,
        verbose=verbose,
        training=training,
        input_names=input_names + omodel.state_input_names,
        output_names=output_names + omodel.state_output_names,
        operator_export_type=operator_export_type,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        dynamic_axes={
            **{i: {0: "batch"} for i in input_names},
            **{o: {0: "batch"} for o in output_names},
            **omodel.state_dynamic_axes,
        },
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        custom_opsets=custom_opsets,
    )


def _shape_list(lst, shape, idx=0):
    if isinstance(shape, int):
        return lst[idx : idx + shape], idx + shape

    assert hasattr(shape, "__len__")
    ret = []
    for s in shape:
        o, idx = _shape_list(lst, s, idx)
        ret.append(o)
    return ret, idx


class OnnxWrapper(torch.nn.Module):
    """Collapses input args and flattens output args.
    This is necessary as the ``dynamic_axes`` arg for
    :py:meth:`torch.onnx.export` doesn't accept nested Tuples.
    Args:
        model: A co.CoModule
    """

    def __init__(self, model: CoModule):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, *states: torch.Tensor):
        shaped_state, _ = _shape_list(states, self.model._state_shape)

        out, next_states = self.model._forward_step(x, shaped_state)
        return (out, *flatten(next_states, remove_none=False))

    @staticmethod
    def _i2o_name(i_name: str) -> str:
        return f"n{i_name}"

    @property
    def state_input_names(self) -> List[str]:
        return [f"s{i}" for i in range(sum(flatten(self.model._state_shape)))]

    @property
    def state_output_names(self) -> List[str]:
        return [self._i2o_name(s) for s in self.state_input_names]

    @property
    def state_dynamic_axes(self) -> Dict[str, List[int]]:
        isdyn = flatten(self.model._dynamic_state_inds)
        ins = {sn: {0: "batch"} for sn, i in zip(self.state_input_names, isdyn) if i}
        outs = {self._i2o_name(k): v for k, v in ins.items()}
        return {**ins, **outs}
