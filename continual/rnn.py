from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence

from .module import CoModule

__all__ = ["RNN", "LSTM", "GRU"]

State = Tuple[Tensor]
LSTMState = Tuple[Tensor, Tensor]


class RNN(CoModule, nn.RNN):
    r"""Applies a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(N, H_{in}, L)` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{out})` containing the initial hidden
          state for each element in the batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(N, H_{out}, L)` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(\text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional RNNs are not supported.

    .. note::
        Contrary to the module version found in torch.nn, this module assumes batch first,
        channel next, and temporal dimension last.

    .. include:: ../cudnn_rnn_determinism.rst

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        rnn = co.RNN(input_size=10, hidden_size=20, num_layers=2)
        #               B, C,  T
        x = torch.randn(1, 10, 16)

        # torch API
        h0 = torch.randn(2, 1, 20)
        output, hn = rnn(x, h0)

        # continual inference API
        rnn.set_state(h0)
        firsts = rnn.forward_steps(x[:,:,:-1])
        last = rnn.forward_step(x[:,:,-1])

        assert torch.allclose(firsts, output[:, :, :-1])
        assert torch.allclose(last, output[:, :, -1])
    """

    _state_shape = 1
    _dynamic_state_inds = [True]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity="tanh",
        bias: bool = True,
        # batch_first: bool = True,  # NB: differs from torch.nn version!
        dropout: float = 0.0,
        # bidirectional: bool = False,  # NB: differs from torch.nn version!
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        # assert (
        #     batch_first
        # ), "`batch_first == False` is not supported for a Continual module"
        # assert (
        #     not bidirectional
        # ), "`bidirectional == True` is not supported for a Continual module"

        nn.RNN.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_from(module: nn.RNN, **kwargs) -> "RNN":
        comodule = RNN(
            **{
                **dict(
                    input_size=module.input_size,
                    hidden_size=module.hidden_size,
                    num_layers=module.num_layers,
                    nonlinearity=module.nonlinearity,
                    bias=module.bias,
                    batch_first=True,
                    dropout=module.dropout,
                    bidirectional=False,
                    device=module._flat_weights[0].device,
                    dtype=module._flat_weights[0].dtype,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            for ours, theirs in zip(comodule._flat_weights, module._flat_weights):
                ours.copy_(theirs)
        return comodule

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        if hasattr(self, "_hidden_state"):
            del self._hidden_state

    def get_state(self) -> Optional[State]:
        if hasattr(self, "_hidden_state"):
            return (self._hidden_state,)

    def set_state(self, state: State):
        if isinstance(state, Tensor):
            state = (state,)
        self._hidden_state = state[0]

    def forward(
        self, input: Union[Tensor, PackedSequence], hx: Optional[Tensor] = None
    ) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        input = input.swapaxes(1, 2)  # B, C, T -> B, T, C
        output, hidden = nn.RNN.forward(self, input, hx)
        output = output.swapaxes(1, 2)  # B, T, C -> B, C, T
        return (output, hidden)

    def forward_step(self, input: Tensor, update_state=True) -> Optional[Tensor]:
        output, next_state = self._forward_step(input, self.get_state())
        if update_state:
            self.set_state(next_state)
        return output

    def _forward_step(
        self, input: Tensor, prev_state: Optional[State] = None
    ) -> Tuple[Tensor, State]:
        input = input.unsqueeze(1)  # B, C -> B, T, C
        hidden_state = (prev_state or (None,))[0]
        output, new_state = nn.RNN.forward(self, input, hidden_state)
        output = output.squeeze(1)  # B, T, C -> B, C
        return output, (new_state,)

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        hidden_state = (self.get_state() or (None,))[0]
        output, new_state = self.forward(input, hidden_state)
        if update_state:
            self.set_state((new_state,))
        return output


class GRU(CoModule, nn.GRU):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(N, H_{in}, L)` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{out})` containing the initial hidden
          state for each element in the batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(N, H_{out}, L)` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(\text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional GRUs are not supported.

    .. note::
        Contrary to the module version found in torch.nn, this module assumes batch first,
        channel next, and temporal dimension last.

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        gru = co.GRU(input_size=10, hidden_size=20, num_layers=2)
        #               B, C,  T
        x = torch.randn(1, 10, 16)

        # torch API
        h0 = torch.randn(2, 1, 20)
        output, hn = gru(x, h0)

        # continual inference API
        gru.set_state(h0)
        firsts = gru.forward_steps(x[:,:,:-1])
        last = gru.forward_step(x[:,:,-1])

        assert torch.allclose(firsts, output[:, :, :-1])
        assert torch.allclose(last, output[:, :, -1])
    """
    _state_shape = 1
    _dynamic_state_inds = [True]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        # batch_first: bool = True,  # NB: differs from torch.nn version!
        dropout: float = 0.0,
        # bidirectional: bool = False,  # NB: differs from torch.nn version!
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        # assert (
        #     batch_first
        # ), "`batch_first == False` is not supported for a Continual module"
        # assert (
        #     not bidirectional
        # ), "`bidirectional == True` is not supported for a Continual module"

        nn.GRU.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_from(module: nn.GRU, **kwargs) -> "GRU":
        comodule = GRU(
            **{
                **dict(
                    input_size=module.input_size,
                    hidden_size=module.hidden_size,
                    num_layers=module.num_layers,
                    bias=module.bias,
                    batch_first=True,
                    dropout=module.dropout,
                    bidirectional=False,
                    device=module._flat_weights[0].device,
                    dtype=module._flat_weights[0].dtype,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            for ours, theirs in zip(comodule._flat_weights, module._flat_weights):
                ours.copy_(theirs)
        return comodule

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        if hasattr(self, "_hidden_state"):
            del self._hidden_state

    def get_state(self) -> Optional[State]:
        if hasattr(self, "_hidden_state"):
            return (self._hidden_state,)

    def set_state(self, state: State):
        if isinstance(state, Tensor):
            state = (state,)
        self._hidden_state = state[0]

    def forward(
        self, input: Union[Tensor, PackedSequence], hx: Optional[Tensor] = None
    ) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        input = input.swapaxes(1, 2)  # B, C, T -> B, T, C
        output, hidden = nn.GRU.forward(self, input, hx)
        output = output.swapaxes(1, 2)  # B, T, C -> B, C, T
        return (output, hidden)

    def forward_step(self, input: Tensor, update_state=True) -> Optional[Tensor]:
        output, next_state = self._forward_step(input, self.get_state())
        if update_state:
            self.set_state(next_state)
        return output

    def _forward_step(
        self, input: Tensor, prev_state: Optional[State] = None
    ) -> Tuple[Tensor, State]:
        input = input.unsqueeze(1)  # B, C -> B, T, C
        hidden_state = (prev_state or (None,))[0]
        output, new_state = nn.GRU.forward(self, input, hidden_state)
        output = output.squeeze(1)  # B, T, C -> B, C
        return output, (new_state,)

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        hidden_state = (self.get_state() or (None,))[0]
        output, new_state = self.forward(input, hidden_state)
        if update_state:
            self.set_state(new_state)
        return output


class LSTM(CoModule, nn.LSTM):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(N, H_{in}, L)` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(N, H_{out}, L)` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(\text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the batch.
        * **c_n**: tensor of shape :math:`(\text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the batch.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If
            ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, proj_size)`.
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
        weight_hr_l[k] : the learnable projection weights of the :math:`\text{k}^{th}` layer
            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was
            specified.
        weight_ih_l[k]_reverse: Analogous to `weight_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hh_l[k]_reverse:  Analogous to `weight_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_ih_l[k]_reverse:  Analogous to `bias_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_hh_l[k]_reverse:  Analogous to `bias_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hr_l[k]_reverse:  Analogous to `weight_hr_l[k]` for the reverse direction.
            Only present when ``bidirectional=True`` and ``proj_size > 0`` was specified.

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional LSTMs are not supported.

    .. note::
        Contrary to the module version found in torch.nn, this module assumes batch first,
        channel next, and temporal dimension last.


    .. include:: ../cudnn_rnn_determinism.rst

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        lstm = co.LSTM(input_size=10, hidden_size=20, num_layers=2)
        #               B, C,  T
        x = torch.randn(1, 10, 16)

        # torch API
        h0 = (torch.randn(2, 1, 20), torch.randn(2, 1, 20))
        output, hn = lstm(x, h0)

        # continual inference API
        lstm.set_state(h0)
        firsts = lstm.forward_steps(x[:,:,:-1])
        last = lstm.forward_step(x[:,:,-1])

        assert torch.allclose(firsts, output[:, :, :-1])
        assert torch.allclose(last, output[:, :, -1])
    """
    _state_shape = 2
    _dynamic_state_inds = [True, True]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        # batch_first: bool = True,  # NB: differs from torch.nn version!
        dropout: float = 0.0,
        # bidirectional: bool = False,  # NB: differs from torch.nn version!
        proj_size=0,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        # assert (
        #     batch_first
        # ), "`batch_first == False` is not supported for a Continual module"
        # assert (
        #     not bidirectional
        # ), "`bidirectional == True` is not supported for a Continual module"

        nn.LSTM.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_from(module: nn.LSTM, **kwargs) -> "LSTM":
        comodule = LSTM(
            **{
                **dict(
                    input_size=module.input_size,
                    hidden_size=module.hidden_size,
                    num_layers=module.num_layers,
                    bias=module.bias,
                    batch_first=True,
                    dropout=module.dropout,
                    bidirectional=False,
                    proj_size=module.proj_size,
                    device=module._flat_weights[0].device,
                    dtype=module._flat_weights[0].dtype,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            for ours, theirs in zip(comodule._flat_weights, module._flat_weights):
                ours.copy_(theirs)
        return comodule

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        if hasattr(self, "_hidden_state"):
            del self._hidden_state
        if hasattr(self, "_cell_state"):
            del self._cell_state

    def get_state(self) -> Optional[LSTMState]:
        if hasattr(self, "_hidden_state") and hasattr(self, "_cell_state"):
            return (self._hidden_state, self._cell_state)

    def set_state(self, state: LSTMState):
        self._hidden_state, self._cell_state = state

    def forward(
        self,
        input: Union[Tensor, PackedSequence],
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Union[Tensor, PackedSequence], LSTMState]:
        input = input.swapaxes(1, 2)  # B, C, T -> B, T, C
        output, hidden = nn.LSTM.forward(self, input, hx)
        output = output.swapaxes(1, 2)  # B, T, C -> B, C, T
        return (output, hidden)

    def forward_step(self, input: Tensor, update_state=True) -> Optional[Tensor]:
        output, next_state = self._forward_step(input, self.get_state())
        if update_state:
            self.set_state(next_state)
        return output

    def _forward_step(
        self, input: Tensor, prev_state: Optional[State] = None
    ) -> Tuple[Tensor, LSTMState]:
        input = input.unsqueeze(1)  # B, C -> B, T, C
        output, new_state = nn.LSTM.forward(self, input, prev_state)
        output = output.squeeze(1)  # B, T, C -> B, C
        return output, new_state

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        output, new_state = self.forward(input, self.get_state())
        if update_state:
            self.set_state(new_state)
        return output
