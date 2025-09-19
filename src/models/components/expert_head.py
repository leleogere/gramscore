import torch
from torch import nn

from models.automata.vectorization_torch import ProblemType
from models.automata.vectorization_torch import TokenTensor


class ExpertHead(nn.Module):
    """
    Expert head for the model. It is responsible to take the output of the decoder and convert
    it into whatever dimensions the token type associated with this head needs.

    :param embedding_dim: The dimension of the embeddings (output of the decoder).
    :param inner_dims: The dimensions of the inner layers.
    :param output_dims: The dimensions of the output layers, one for each dimension required.
    """

    def __init__(self, embedding_dim: int, inner_dims: list[int], output_dims: dict[str, int]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.inner_dims = inner_dims
        self.output_dims = output_dims

        self.layers = nn.Sequential(
            nn.Linear(self.embedding_dim, self.inner_dims[0]),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(self.inner_dims[i], self.inner_dims[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(self.inner_dims) - 1)
            ],
        )
        self.output_layers = nn.ModuleDict(
            {
                dimension_name: nn.Linear(self.inner_dims[-1], output_dim)
                for dimension_name, output_dim in output_dims.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the expert head.

        :param x: The input tensor.
        :return: A dictionary of output tensors, one for each dimension.
        """
        x = self.layers(x)
        outputs = {
            dimension_name: output_layer(x)
            for dimension_name, output_layer in self.output_layers.items()
        }
        return outputs

    def get_losses(self, include_token_type: bool = False) -> dict[str, nn.Module]:
        losses = {}
        for dim_name in self.output_dims.keys():
            if not include_token_type and dim_name == "token_type":
                continue
            match TokenTensor.PROBLEM_TYPES[dim_name]:
                case ProblemType.CLASSIFICATION:
                    losses[dim_name] = nn.CrossEntropyLoss()
                case ProblemType.REGRESSION:
                    losses[dim_name] = nn.MSELoss()
                case _:
                    raise ValueError(f"Unsupported problem type for dimension {dim_name}")
        return losses
