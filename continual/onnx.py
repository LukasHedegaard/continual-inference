from typing import Dict, List

import torch

from continual import CoModule
from continual.utils import flatten


def export(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=None,
    opset_version=None,
    do_constant_folding=True,
    example_outputs=None,
    strip_doc_string=True,
    keep_initializers_as_inputs=None,
    custom_opsets=None,
    enable_onnx_checker=True,
    use_external_data_format=False,
):
    r"""
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported;
    at the moment, it supports a limited set of dynamic models (e.g., RNNs.)

    Args:
        model (co.CoModule): the model to be exported.
        args (tuple of arguments or torch.Tensor, a dictionary consisting of named arguments (optional)):
            a dictionary to specify the input to the corresponding named parameter:
            - KEY: str, named parameter
            - VALUE: corresponding input
            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS or torch.Tensor::

                "args = (x, y, z)"

            The inputs to the model, e.g., such that ``model(*args)`` is a valid invocation
            of the model. Any non-Tensor arguments will be hard-coded into the exported model;
            any Tensor arguments will become inputs of the exported model, in the order they
            occur in args. If args is a Tensor, this is equivalent to having
            called it with a 1-ary tuple of that Tensor.

            2. A TUPLE OF ARGUEMENTS WITH A DICTIONARY OF NAMED PARAMETERS::

                "args = (x,
                        {
                        'y': input_y,
                        'z': input_z
                        })"

            The inputs to the model are structured as a tuple consisting of
            non-keyword arguments and the last value of this tuple being a dictionary
            consisting of named parameters and the corresponding inputs as key-value pairs.
            If certain named argument is not present in the dictionary, it is assigned
            the default value, or None if default value is not provided.

            Cases in which an dictionary input is the last input of the args tuple
            would cause a conflict when a dictionary of named parameters is used.
            The model below provides such an example.

                class Model(torch.nn.Module):
                    def forward(self, k, x):
                        ...
                        return x

                m = Model()
                k = torch.randn(2, 3)
                x = {torch.tensor(1.): torch.randn(2, 3)}

                In the previous iteration, the call to export API would look like

                    torch.onnx.export(model, (k, x), 'test.onnx')

                This would work as intended. However, the export function
                would now assume that the `x` input is intended to represent the optional
                dictionary consisting of named arguments. In order to prevent this from being
                an issue a constraint is placed to provide an empty dictionary as the last
                input in the tuple args in such cases. The new call would look like this.

                    torch.onnx.export(model, (k, x, {}), 'test.onnx')

        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        training (enum, default TrainingMode.EVAL):
            TrainingMode.EVAL: export the model in inference mode.
            TrainingMode.PRESERVE: export the model in inference mode if model.training is
            False and to a training friendly mode if model.training is True.
            TrainingMode.TRAINING: export the model in a training friendly mode.
        input_names(list of strings, default empty list): names to assign to the
            input nodes of the graph, in order
        output_names(list of strings, default empty list): names to assign to the
            output nodes of the graph, in order
        operator_export_type (enum, default OperatorExportTypes.ONNX):
            OperatorExportTypes.ONNX: All ops are exported as regular ONNX ops
            (with ONNX namespace).
            OperatorExportTypes.ONNX_ATEN: All ops are exported as ATen ops
            (with aten namespace).
            OperatorExportTypes.ONNX_ATEN_FALLBACK: If an ATen op is not supported
            in ONNX or its symbolic is missing, fall back on ATen op. Registered ops
            are exported to ONNX regularly.
            Example graph::

                graph(%0 : Float)::
                  %3 : int = prim::Constant[value=0]()
                  %4 : Float = aten::triu(%0, %3) # missing op
                  %5 : Float = aten::mul(%4, %0) # registered op
                  return (%5)

            is exported as::

                graph(%0 : Float)::
                  %1 : Long() = onnx::Constant[value={0}]()
                  %2 : Float = aten::ATen[operator="triu"](%0, %1)  # missing op
                  %3 : Float = onnx::Mul(%2, %0) # registered op
                  return (%3)

            In the above example, aten::triu is not supported in ONNX, hence
            exporter falls back on this op.
            OperatorExportTypes.RAW: Export raw ir.
            OperatorExportTypes.ONNX_FALLTHROUGH: If an op is not supported
            in ONNX, fall through and export the operator as is, as a custom
            ONNX op. Using this mode, the op can be exported and implemented by
            the user for their runtime backend.
            Example graph::

                graph(%x.1 : Long(1, strides=[1]))::
                  %1 : None = prim::Constant()
                  %2 : Tensor = aten::sum(%x.1, %1)
                  %y.1 : Tensor[] = prim::ListConstruct(%2)
                  return (%y.1)

            is exported as::

                graph(%x.1 : Long(1, strides=[1]))::
                  %1 : Tensor = onnx::ReduceSum[keepdims=0](%x.1)
                  %y.1 : Long() = prim::ListConstruct(%1)
                  return (%y.1)

            In the above example, prim::ListConstruct is not supported, hence
            exporter falls through.

        opset_version (int, default is 9): by default we export the model to the
            opset version of the onnx submodule. Since ONNX's latest opset may
            evolve before next stable release, by default we export to one stable
            opset version. Right now, supported stable opset version is 9.
            The opset_version must be _onnx_main_opset or in _onnx_stable_opsets
            which are defined in torch/onnx/symbolic_helper.py
        do_constant_folding (bool, default False): If True, the constant-folding
            optimization is applied to the model during export. Constant-folding
            optimization will replace some of the ops that have all constant
            inputs, with pre-computed constant nodes.
        example_outputs (tuple of Tensors, list of Tensors, Tensor, int, float, bool, default None):
            Model's example outputs being exported. 'example_outputs' must be provided when exporting
            a ScriptModule or TorchScript Function. If there is more than one item, it should be passed
            in tuple format, e.g.: example_outputs = (x, y, z). Otherwise, only one item should
            be passed as the example output, e.g. example_outputs=x.
            example_outputs must be provided when exporting a ScriptModule or TorchScript Function.
        strip_doc_string (bool, default True): if True, strips the field
            "doc_string" from the exported model, which information about the stack
            trace.
        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.

            This may allow for better optimizations (such as constant folding
            etc.) by backends/runtimes that execute these graphs. If
            unspecified (default None), then the behavior is chosen
            automatically as follows. If operator_export_type is
            OperatorExportTypes.ONNX, the behavior is equivalent to setting
            this argument to False. For other values of operator_export_type,
            the behavior is equivalent to setting this argument to True. Note
            that for ONNX opset version < 9, initializers MUST be part of graph
            inputs. Therefore, if opset_version argument is set to a 8 or
            lower, this argument will be ignored.
        custom_opsets (dict<string, int>, default empty dict): A dictionary to indicate
            custom opset domain and version at export. If model contains a custom opset,
            it is optional to specify the domain and opset version in the dictionary:
            - KEY: opset domain name
            - VALUE: opset version
            If the custom opset is not provided in this dictionary, opset version is set
            to 1 by default.
        enable_onnx_checker (bool, default True): If True the onnx model checker will be run
            as part of the export, to ensure the exported model is a valid ONNX model.
        use_external_data_format (bool, default False): If True, then the model is exported
            in ONNX external data format, in which case some of the model parameters are stored
            in external binary files and not in the ONNX model file itself. See link for format
            details:
            https://github.com/onnx/onnx/blob/8b3f7e2e7a0f2aba0e629e23d89f07c7fc0e6a5e/onnx/onnx.proto#L423
            Also, in this case,  argument 'f' must be a string specifying the location of the model.
            The external binary files will be stored in the same location specified by the model
            location 'f'. If False, then the model is stored in regular format, i.e. model and
            parameters are all in one file. This argument is ignored for all export types other
            than ONNX.
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
        example_outputs=example_outputs,
        strip_doc_string=strip_doc_string,
        dynamic_axes={
            **{i: {0: "batch"} for i in input_names},
            **{o: {0: "batch"} for o in output_names},
            **omodel.state_dynamic_axes,
        },
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        custom_opsets=custom_opsets,
        enable_onnx_checker=enable_onnx_checker,
        use_external_data_format=use_external_data_format,
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
