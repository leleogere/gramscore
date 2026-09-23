import torch
from torch import nn
import math
from loguru import logger


class ContinuousPositionalEncoding1D(nn.Module):

    def __init__(
        self,
        d_model: int,
        shortest_period: float = 0.1,
        longest_period: float = 300.0,
    ):
        """
        Create a continuous positional encoding for a sequence.
        This module is equivalent to the classical 10_000-based absolute positional encoding for
        shortest_period=2*pi and longest_period=10_000*2*pi.

        :param d_model: Dimension of the model.
        :param shortest_period: Period of the shortest sinusoid (controls the shortest amount of
            time that the model can distinguish).
        :param longest_period: Period of the longest sinusoid (controls the longest amount of time
            that the model can distinguish).
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        self._d_model = d_model
        self._shortest_period = shortest_period
        self._longest_period = longest_period

        # Compute the div_term so that the sinusoids have the correct period (between
        # smallest_period_duration and largest_period_duration).
        div_term = (
            math.tau
            / shortest_period
            * (
                torch.exp(
                    2
                    * torch.arange(0, d_model // 2)
                    * (-math.log(longest_period / shortest_period) / (d_model - 2))
                )  # The d-2 above comes from i in [0, d/2 - 1], therefore 2i in [0, d - 2]
            )
        )
        # print(f"{div_term.shape = }\n{div_term}\n")
        self.register_buffer("div_term", div_term)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def shortest_period(self) -> float:
        return self._shortest_period

    @property
    def longest_period(self) -> float:
        return self._longest_period

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        :param positions: Tensor of any shape containing positions.
        :return: Positional Encoding Matrix of shape (*positions.shape, d_model)
        """
        if (oob := positions.max()) > self.longest_period:
            logger.warning(
                f"Positions should be in [0, {self.longest_period}] to be properly encoded. "
                f"Got max position {oob}, consider increasing longest_period."
            )
        pe = torch.zeros((*positions.shape, self.d_model), device=positions.device)
        positions = positions.unsqueeze(-1)
        # print(f"{positions.shape = }\n{pe.shape}\n")
        scaled_position = positions * self.div_term
        sentinel = self.d_model // 2
        pe[..., :sentinel] = torch.sin(scaled_position)
        pe[..., sentinel:] = torch.cos(scaled_position)
        return pe

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"shortest_period={self.shortest_period}, "
            f"longest_period={self.longest_period}"
        )


# class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
#     """This module produces sinusoidal positional embeddings of any length."""
#
#     def __init__(
#         self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
#     ) -> None:
#         super().__init__(num_positions, embedding_dim)
#
#     def _init_weight(self):
#         """
#         Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
#         the 2nd half of the vector. [dim // 2:]
#         """
#         n_pos, dim = self.weight.shape
#         position_enc = np.array(
#             [
#                 [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
#                 for pos in range(n_pos)
#             ]
#         )
#         out = torch.empty(n_pos, dim, dtype=self.weight.dtype, requires_grad=False)
#         sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
#         out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
#         out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
#         self.weight = nn.Parameter(out, requires_grad=False)
#
#     @torch.no_grad()
#     def forward(
#         self,
#         input_ids_shape: torch.Size,
#         past_key_values_length: int = 0,
#         position_ids: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """`input_ids_shape` is expected to be [bsz x seqlen]."""
#         if position_ids is None:
#             bsz, seq_len = input_ids_shape[:2]
#             position_ids = torch.arange(
#                 past_key_values_length,
#                 past_key_values_length + seq_len,
#                 dtype=torch.long,
#                 device=self.weight.device,
#             )
#         return super().forward(position_ids)


if __name__ == '__main__':
    pe = ContinuousPositionalEncoding1D(
        d_model=64,
        shortest_period=0.01,
        longest_period=600.0,
    )
    positions = torch.linspace(0, 1, steps=10000)
    print(f"{positions.shape = }")
    out = pe(positions)
    print(f"{out.shape = }")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=pe.d_model, figsize=(10, 2 * pe.d_model))
    if pe.d_model == 1:
        axes = [axes]
    for j in range(pe.d_model):
        axes[j].plot(positions.cpu(), out[:, j].cpu())
        axes[j].set_title(f"Dimension {j}")
        axes[j].grid(True)
        axes[j].set_ylim([-1.1, 1.1])
    plt.tight_layout()
    plt.show()
