import torch
from torch import nn

from models.automata.vectorization_torch import TokenTensor


class MIDIEmbedding(nn.Module):
    """
    Construct the embeddings from MIDI token streams.
    The embedding in simply the sum of the embeddings of a linear projection of each stream into
    the embedding dimension (followed by layer norm and dropout).
    """

    def __init__(
        self,
        stream_dimensions: dict[str, int],
        embedding_dimension: int,
        layer_norm_eps: float,
        dropout_prob: float,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.stream_embeddings = nn.ParameterDict(
            {k: nn.Linear(v, self.embedding_dimension) for k, v in stream_dimensions.items()}
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dimension, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Projects fixed-size embeddings into parallel token streams representing MIDI data.

        :param embeddings: Dictionary of tensors of shape (batch, n_notes, embedding_dim)

        :return: Unique embedding tensor of shape (batch, n_notes, embedding_dim)
        """
        # Apply layer norm and dropout to the input embeddings
        stream_embeddings = {k: v(embeddings[k]) for k, v in self.stream_embeddings.items()}
        embeddings = torch.sum(torch.stack(list(stream_embeddings.values())), dim=0)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ScoreTokenEmbedding(nn.Module):
    """
    Project embeddings to compound tokens for score data.
    Each token (dimension TokenTensor.TOTAL_DIMENSIONS) is projected into an embedding space,
    followed by a layer norm and dropout.
    """

    def __init__(
        self,
        embedding_dimension: int,
        layer_norm_eps: float,
        dropout_prob: float,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Linear(TokenTensor.TOTAL_DIMENSIONS, self.embedding_dimension)
        self.layer_norm = nn.LayerNorm(self.embedding_dimension, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, tokens: TokenTensor) -> torch.Tensor:
        """Projects fixed-size embeddings into parallel token streams representing MIDI data.

        :param tokens: TokenTensor object containing the vectorized tokens

        :return: Embedding tensor of shape (batch, n_notes, embedding_dim)
        """
        # Apply layer norm and dropout to the input embeddings
        embeddings = self.embedding(tokens)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
