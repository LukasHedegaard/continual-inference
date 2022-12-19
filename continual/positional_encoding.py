from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

import continual as co

__all__ = ["RecyclingPositionalEncoding"]


class CyclicPositionalEncoding(nn.Module):
    """Cyclic Positional Encoding as proposed by Ma et al. in
    "Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer"
    https://arxiv.org/abs/2110.02544
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, mean_pooling=True):
        super(CyclicPositionalEncoding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        skip_base = np.power(num_embeddings, 1 / (embedding_dim // 2))
        skip_set = np.linspace(
            skip_base, num_embeddings, embedding_dim // 2, dtype="int"
        )
        x = np.zeros((num_embeddings, embedding_dim))

        def basesin(x, omiga, fai=0):
            T = 2 * np.pi / omiga
            return np.sin(omiga * np.abs(np.mod(x, 2 * T) - T) + fai)

        def basecos(x, omiga, fai=0):
            T = 2 * np.pi / omiga
            return np.cos(omiga * np.abs(np.mod(x, 2 * T) - T) + fai)

        for i in range(embedding_dim):
            # see Appendix B
            skip = (
                skip_set[i // 3 * 3 + 1]
                if (i // 3 * 3 + 1) < (embedding_dim // 2)
                else skip_set[-1]
            )

            # get z(i) in the paper (via longer_pattern)
            if num_embeddings > skip:
                longer_pattern = np.arange(
                    0, np.ceil((num_embeddings) / skip) * skip + 0.01, 0.01
                )
            else:
                longer_pattern = np.arange(0, num_embeddings + 0.01, 0.01)
                skip = num_embeddings

            num = len(longer_pattern) - 1
            omiga = 2 * np.pi / skip

            # see Appendix B
            fai = (
                0
                if i <= (embedding_dim // 2)
                else 2 * np.pi * ((-i + (embedding_dim // 2)) / (embedding_dim // 2))
            )

            # Eq. (4) in the paper
            if i % 2 == 1:
                x[:, i] = basecos(longer_pattern, omiga, fai)[
                    np.linspace(0, num, num_embeddings + 1, dtype="int")
                ][:num_embeddings]
            else:
                x[:, i] = basesin(longer_pattern, omiga, fai)[
                    np.linspace(0, num, num_embeddings + 1, dtype="int")
                ][:num_embeddings]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # Averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(num_embeddings)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + num_embeddings) % num_embeddings
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1.0 / time * pattern_sum - pattern.mean(0)
        self.register_buffer("pattern", pattern)

    def forward(self, input: Tensor) -> Tensor:
        return self.pattern[input]


State = Tuple[Tensor]


class RecyclingPositionalEncoding(co.CoModule, nn.Module):
    """Recycling Positional Encoding with learned or static weights.

    Recycling Positional Encoding were proposed by Hedegaard et al. in
    "Continual Transformers: Redundancy-Free Attention for Online Inference"
    https://arxiv.org/abs/2201.06268 (paper) https://www.youtube.com/watch?v=gy802Tlp-eQ (video).

    When static encoding is selected, the module employs "Cyclic Positional Encoding" as
    proposed by Ma et al. in
    "Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer"
    https://arxiv.org/abs/2110.02544.

    Args:
        embed_dim: dimensionality of positional embeddings.
        num_embeds: number of embeddings to recycle among.
        learned: whether embeddings should be learned or static sinusoidal
        forward_update_index_steps: the number of index steps to offset the encoding query with
            each time ``forward`` is called. This ensures that positional encodings have a
            new starting position at each call.

    Examples::

        pe = RecyclingPositionalEncoding(
            embed_dim=10,
            num_embeds=16 * 2 - 1,
            forward_update_index_steps=0
        )
        x = torch.zeros((1, 10, 16))  # (B, C, T)

        o_forward = pe.forward(x)
        o_forward_steps = pe.forward_steps(x[:, :, :-1])
        o_forward_step = pe.forward_step(x[:, :, -1])

        assert torch.equal(o_forward[:, :, :-1], o_forward_steps)
        assert torch.equal(o_forward[:, :, -1], o_forward_step)

    """

    _state_shape = 1
    _dynamic_state_inds = [False]

    def __init__(
        self,
        embed_dim: int,
        num_embeds: int,
        learned: bool = True,
        forward_update_index_steps: int = 1,
    ):
        nn.Module.__init__(self)
        self.pe = {True: nn.Embedding, False: CyclicPositionalEncoding}[learned](
            num_embeds, embed_dim
        )
        self.register_buffer("state_index", torch.tensor(0), persistent=False)
        self.forward_update_index_steps = forward_update_index_steps

    def forward(self, input: Tensor, update_index_steps: int = None) -> Tensor:
        T = input.shape[2]
        assert T <= self.pe.num_embeddings
        position_ids = (
            torch.arange(T, device=input.device).unsqueeze(0) + self.state_index
        ) % self.pe.num_embeddings

        index_update = (
            self.forward_update_index_steps
            if update_index_steps is None
            else update_index_steps
        )
        self.state_index = (self.state_index + index_update) % self.pe.num_embeddings

        position_embeddings = self.pe(position_ids).transpose(1, 2)
        return input + position_embeddings

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        return self.forward(
            input, update_index_steps=input.shape[2] if update_state else 0
        )

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        output = input + self.pe(self.state_index.unsqueeze(0))

        if update_state:
            self.state_index = (self.state_index + 1) % self.pe.num_embeddings
        return output

    def _forward_step(
        self, input: Tensor, prev_state: Optional[State] = None
    ) -> Tuple[Tensor, State]:
        if prev_state is None:
            state_index = self.init_state()[0]
        else:
            state_index = prev_state[0]
        output = input + self.pe(state_index.unsqueeze(0))

        state_index = (state_index + 1) % self.pe.num_embeddings
        return output, (state_index,)

    def init_state(self) -> State:
        self.state_index = torch.tensor(0)
        return (self.state_index,)

    def clean_state(self):
        self.state_index = torch.tensor(0)
