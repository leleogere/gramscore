from collections.abc import Mapping

import torch
import torch.nn as nn

from models.automata.vectorization_torch import TokenTensor
from models.components.embedding import MIDIEmbedding
from models.components.embedding import ScoreTokenEmbedding
from models.components.expert_head import ExpertHead
from models.components.roformer import Roformer
from models.automata import tokens as tk


class TokensTranscriberModel(nn.Module):
    def __init__(
        self,
        stream_dimensions: dict[str, int],
        roformer: Roformer | None,
        expert_inner_dims: list[int],
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

        self.embedding_dim = self.roformer.encoder.config.hidden_size
        self.layer_norm_eps = self.roformer.encoder.config.layer_norm_eps
        self.hidden_dropout_prob = self.roformer.encoder.config.hidden_dropout_prob

        self.midi_embedding = MIDIEmbedding(
            stream_dimensions=stream_dimensions,
            embedding_dimension=self.embedding_dim,
            layer_norm_eps=self.layer_norm_eps,
            dropout_prob=self.hidden_dropout_prob,
        )

        self.score_embedding = ScoreTokenEmbedding(
            embedding_dimension=self.embedding_dim,
            layer_norm_eps=self.layer_norm_eps,
            dropout_prob=self.hidden_dropout_prob,
        )

        self.token_type_router = nn.Linear(self.embedding_dim, len(TokenTensor.TOKEN_INDEXER))

        self.expert_inner_dims = expert_inner_dims
        self.expert_heads: Mapping[str, ExpertHead] = nn.ModuleDict(
            {
                token_type.as_str(): ExpertHead(
                    embedding_dim=self.embedding_dim,
                    inner_dims=self.expert_inner_dims,
                    output_dims={
                        dimension_name: TokenTensor.DIMS[dimension_name]
                        for dimension_name in TokenTensor.TOKEN_TYPE_TO_DIMS[token_type]
                        if dimension_name != "token_type"
                    },
                )
                for token_type in tk.TOKEN_TYPES  # TODO: Remove Tokens without attributes
                if token_type
                not in [
                    tk.StartOfScore,  # Not predicted
                    tk.EndOfScore,  # No attributes
                ]
            }
        )

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

    def forward_router(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        return self.token_type_router(decoder_outputs)

    def forward_expert(
        self,
        token_type: tk.AnyToken,
        decoder_outputs: torch.Tensor,  # B * T * E
    ) -> dict[str, torch.Tensor]:  # i: B * T * dim[i]
        """
        Foward the tensor though one specific expert head. Note that all tokens will flow though
        the expert, you only need some indices to pass though this expert, make sure to filter them
        out of decoder_outputs before.
        """
        expert_head: ExpertHead = self.expert_heads[token_type]
        expert_outputs = expert_head.forward(decoder_outputs)
        return expert_outputs

    def forward_experts(
        self,
        decoder_outputs: torch.Tensor,  # B * T * E
        token_type_masks: dict[str, torch.Tensor],  # B * T
        convert_to_tokentensor: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]] | TokenTensor:
        # Check that the shapes make sense
        if not all(mask.shape == decoder_outputs.shape[:2] for mask in token_type_masks.values()):
            raise ValueError(
                f"Dimensions mismatch: {decoder_outputs.shape[:2]=} != "
                f"{[mask.shape for mask in token_type_masks.values()]}"
            )

        # Split the decoder_outputs into experts
        # TODO: change to dict comprehension
        expert_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for token_type in self.expert_heads:
            expert: ExpertHead = self.expert_heads[token_type]
            involved_decoder_outputs = decoder_outputs[token_type_masks[token_type]]
            expert_output = expert.forward(involved_decoder_outputs)
            expert_outputs[token_type] = expert_output

        if convert_to_tokentensor:
            return TokenTensor.build_from_dict(
                type_dim_values_dict=expert_outputs,
                token_type_masks=token_type_masks,
            )
        else:
            return expert_outputs
