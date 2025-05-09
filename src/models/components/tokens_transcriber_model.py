import torch
import torch.nn as nn

from models.automata.vectorization_torch import TokenTensor
from models.components.embedding import MIDIEmbedding
from models.components.embedding import ScoreTokenEmbedding
from models.components.roformer import Roformer


class TokensTranscriberModel(nn.Module):
    def __init__(
        self,
        stream_dimensions: dict[str, int],
        roformer: Roformer | None,
    ):
        super().__init__()

        if roformer is None:
            self.roformer = Roformer(
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                intermediate_size=1536,
                hidden_dropout_prob=0.1,
                layer_norm_eps=1e-12,
            )
        else:
            self.roformer = roformer
        # TODO: move embedding logic from Roformer to here
        # The roformer should only be responsible for the transformer part, taking
        # the embeddings as input, and returning the output embeddings
        # The embedding logic, router logic and token type attribute predictors logic
        # should all lay in this module

        hidden_size = self.roformer.encoder.config.hidden_size
        layer_norm_eps = self.roformer.encoder.config.layer_norm_eps
        hidden_dropout_prob = self.roformer.encoder.config.hidden_dropout_prob

        self.midi_embedding = MIDIEmbedding(
            stream_dimensions=stream_dimensions,
            embedding_dimension=hidden_size,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )

        self.score_embedding = ScoreTokenEmbedding(
            embedding_dimension=hidden_size,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )

        self.token_type_router = nn.Linear(hidden_size, len(TokenTensor.TOKEN_INDEXER))

    def forward_router(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        return self.token_type_router(decoder_outputs)

    def forward_encoder(
        self,
        input_streams: dict[str, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        midi_embeddings = self.midi_embedding.forward(input_streams)
        outputs_encoder = self.roformer.forward_enc(
            midi_embeddings=midi_embeddings, attention_mask=attention_mask
        )
        return outputs_encoder

    def forward_decoder(
        self,
        input_tokens: TokenTensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        score_embeddings = self.score_embedding.forward(input_tokens)
        outputs_decoder = self.roformer.forward_dec(
            score_embeddings=score_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return outputs_decoder
