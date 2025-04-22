import itertools

import torch
import torch.nn as nn
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from models.automata.vectorization_torch import TokenTensor
import models.automata.tokens as tk
from models.components.roformer import Roformer


class TokensTranscriberModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.embedding_size = 512

        # PERFORMANCE PROJECTOR
        self.stream_dimensions = {
            "pitch": 128,
            "onset": 200,
            "duration": 200,
            "velocity": 8,
        }
        self.streams_projectors = nn.ModuleDict(
            {
                stream: nn.Linear(dim, self.embedding_size)
                for stream, dim in [
                    ("pitch", 128),
                    ("onset", 200),
                    ("duration", 200),
                    ("velocity", 8),
                ]  # pitch/onset/duration/velocity (as in Beyer's paper)
            }
        )
        # Question: should the PE be placed before or after the layer norm?
        # self.performance_positional_encoding = RotaryPositionalEmbeddings(dim=self.embedding_size)
        self.performance_positional_encoding = Summer(PositionalEncoding1D(self.embedding_size))
        self.layer_norm = nn.LayerNorm(self.embedding_size)

        # SCORE PROJECTOR
        self.score_projector = nn.Linear(TokenTensor.TOTAL_DIMENSIONS, self.embedding_size)
        self.score_layer_norm = nn.LayerNorm(self.embedding_size)

        # TRANSFORMER
        self.transformer: nn.Transformer = nn.Transformer(
            d_model=self.embedding_size,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=3072,
            batch_first=True,
        )

        # TOKEN TYPE CLASSIFIER
        self.router = nn.Sequential(
            nn.Linear(self.embedding_size, len(TokenTensor.TOKEN_INDEXER)),
        )

        # # TOKEN-SPECIFIC PREDICTORS
        # self.final_predictors = nn.ModuleDict(
        #     {
        #         "StartOfScore": None,
        #         "EndOfScore": None,
        #         "Bar": BarSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "KeySignature": KeySignatureSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "Clef": ClefSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "Rest": RestSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "Note": NoteSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "ChordStart": ChordStartSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "ChordEnd": ChordEndSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "TupletStart": TupletStartSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #         "TupletEnd": TupletEndSpecificPredictor(TokenTensor.TOTAL_DIMENSIONS),
        #     }
        # )
        # assert len(self.final_predictors) == len(TokenTensor.TOKEN_INDEXER)

        # LOSSES
        self.token_type_loss = nn.NLLLoss()
        # self.specific_losses = nn.ModuleDict(
        #     {token_type: predictor.get_loss() for token_type, predictor in self.final_predictors}
        # )

    # def get_loss(self):
    #     class Loss(nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.token_type_loss = nn.NLLLoss()
    #             self.predictors_loss = {
    #                 name: predictor.get_loss() for name, predictor in self.final_predictors.items()
    #             }
    #             self.predictors_loss_idx = {i: v for v in enumerate(self.predictors_loss.values())}
    #
    #         def forward(
    #             self,
    #             token_type_input: torch.Tensor,  # B * N_TOK_TYPE * M
    #             token_type_target: torch.Tensor,  # same
    #             specific_input: torch.Tensor | None,  # B * TOTAL_DIM * M
    #             specific_target: torch.Tensor | None,  # same
    #         ):
    #             token_type_loss_value = self.token_type_loss(token_type_input, token_type_target)
    #             specific_loss = self.predictors_loss[specific_input]

    def forward(
        self,
        src_streams: dict[str, torch.Tensor],  # batch * num_notes * feats = B * N * F
        tgt_tokens: torch.Tensor,  # batch * num_tokens * token_dim = B * T * K
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
    ):
        """
        Forward pass of the model.

        Variables used for shapes:
        B = batch_size (src_streams[x].shape[0], tgt_tokens.shape[0])
        N = number of notes in performance (src_streams.shape[1])
        T = number of tokens in score (tgt_tokens.shape[1])
        E = embedding dimension (512)
        D = token size (TokenTensor.TOTAL_DIMENSIONS)
        K = number of token types (len(TokenTensor.TOKEN_INDEXER))
        """
        # Project the performance streams into a merged 512-dim space
        stream_projected = torch.stack(
            [self.streams_projectors[stream](values) for stream, values in src_streams.items()],
            dim=0,
        )  # 4 * B * N * E
        src_emb = torch.sum(stream_projected, dim=0)  # B * N * E
        # Question: should the PE be placed before or after the layer norm?
        src_emb = self.performance_positional_encoding(src_emb)  # B * N * E
        src_emb = self.layer_norm(src_emb)  # B * N * E
        assert src_emb.shape[2] == self.embedding_size
        assert src_emb.shape[0] == tgt_tokens.shape[0]

        # Project the tokens into a 512-dim space
        tgt_emb = self.score_projector(tgt_tokens)  # B * T * E
        tgt_emb = self.score_layer_norm(tgt_emb)
        assert tgt_emb.shape[2] == self.embedding_size

        # Pass the performance and the score through the transformer
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_emb.shape[1],
            device=tgt_emb.device,
            dtype=tgt_emb.dtype,
        )  # T * T
        # print(f"{src_emb.dtype = }")
        # print(f"{tgt_emb.dtype = }")
        # print(f"{tgt_mask.dtype = }")
        output_transformer = self.transformer.forward(
            src=src_emb,
            tgt=tgt_emb,
            tgt_is_causal=True,
            tgt_mask=tgt_mask,
        )  # B * T * E

        # Predict the token types with the router
        logits_router = self.router(output_transformer)  # B * T * K

        # Pass the output of the transformer through the token-specific predictors
        # TODO

        # For now only try to predict token types
        return logits_router


class TokenSpecificPreditor(nn.Module):
    def __init__(self, in_dim: int, dimensions_names: list[str], token_type_idx: int):
        assert all(dim in TokenTensor.DIMS for dim in dimensions_names), "Invalid dimensions"

        super().__init__()

        self.token_type_idx = token_type_idx
        self.in_dim = in_dim
        self.dimensions = {dim: TokenTensor.DIMS[dim] for dim in dimensions_names}
        self.dimension_dims = {dim: TokenTensor.DIMS[dim] for dim in dimensions_names}
        self.dimension_cum_dims = {
            k: v
            for k, v in zip(
                self.dimensions, itertools.accumulate(self.dimension_dims.values(), initial=0)
            )
        }
        self.out_dim = sum(self.dimension_dims.values())
        self.feed_forward = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim),
        )

    @staticmethod
    def empty_tensor(shape: tuple[int]):
        """
        Returns an empty tensor with the given shape. The result will be a tensor of shape
        (*shape, TokenTensor.TOTAL_DIMENSIONS) with all values set to 0.
        """
        total_shape = (*shape, TokenTensor.TOTAL_DIMENSIONS)
        return TokenTensor(torch.zeros(total_shape))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.
        """
        x = self.feed_forward(x)
        x = self.predictor_specific_post_processing(x)
        return x

    def predictor_specific_post_processing(self, x: torch.Tensor):
        """
        Post-processing specific to the predictor.
        """
        raise NotImplementedError()

    @classmethod
    def get_loss(cls):
        """
        Returns the loss function for the predictor.
        """
        raise NotImplementedError()

    def to_full_vector(self, partial_vector: torch.Tensor) -> TokenTensor:
        """
        Converts a partial vector to a full vector.
        """
        # Create an empty vector
        full_vector = self.empty_tensor(partial_vector.shape[:-1])
        # Add the token type at the correct position
        token_type_idx = self.token_type_idx
        full_vector[TokenTensor.CUM_DIMS["token_type"] + token_type_idx] = 1
        # Add each dimension at the correct position
        for dim, local_index in self.dimensions.items():
            full_index = TokenTensor.CUM_DIMS[dim]
            full_vector[..., full_index] = partial_vector[..., local_index]
        return full_vector


class BarSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(
            in_dim,
            ["bar_numerator", "bar_denominator", "bar_is_nominal"],
            TokenTensor.TOKEN_INDEXER[tk.Bar],
        )
        self.bar_denominator_activation = nn.Softmax(dim=-1)
        self.bar_is_nominal_activation = nn.Sigmoid()

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Bar numerator -> Nothing to do
        # Bar denominator -> Softmax
        x[self.dimensions["bar_denominator"]] = self.bar_denominator_activation(
            x[self.dimensions["bar_denominator"]]
        )
        # Bar is nominal -> Sigmoid
        x[self.dimensions["bar_is_nominal"]] = self.bar_is_nominal_activation(
            x[self.dimensions["bar_is_nominal"]]
        )
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bar_numerator_regression = nn.MSELoss()
                self.bar_denominator_classification = nn.NLLLoss()  # No CE as softmax already done
                self.bar_is_nominal_classification = nn.BCELoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                bar_numerator_loss = self.bar_numerator_regression(
                    input[self.dimensions["bar_numerator"]],
                    target[self.dimensions["bar_numerator"]],
                )
                bar_denominator_loss = self.bar_denominator_classification(
                    input[self.dimensions["bar_denominator"]],
                    target[self.dimensions["bar_denominator"]],
                )
                bar_is_nominal_loss = self.bar_is_nominal_classification(
                    input[self.dimensions["bar_is_nominal"]],
                    target[self.dimensions["bar_is_nominal"]],
                )
                return bar_numerator_loss + bar_denominator_loss + bar_is_nominal_loss

        return Loss


class KeySignatureSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(in_dim, ["key_signature"], TokenTensor.TOKEN_INDEXER[tk.KeySignature])
        self.key_signature_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Key signature -> Softmax
        x[self.dimensions["key_signature"]] = self.key_signature_activation(
            x[self.dimensions["key_signature"]]
        )
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.key_signature_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                key_signature_loss = self.key_signature_classification(
                    input[self.dimensions["key_signature"]],
                    target[self.dimensions["key_signature"]],
                )
                return key_signature_loss

        return Loss


class ClefSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(in_dim, ["clef", "staff_idx"], TokenTensor.TOKEN_INDEXER[tk.Clef])
        self.clef_activation = nn.Sigmoid(dim=-1)
        self.staff_idx_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Clef -> Sigmoid
        x[self.dimensions["clef"]] = self.clef_activation(x[self.dimensions["clef"]])
        # Staff index -> Softmax
        x[self.dimensions["staff_idx"]] = self.staff_idx_activation(x[self.dimensions["staff_idx"]])
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.clef_classification = nn.BCELoss()
                self.staff_idx_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                clef_loss = self.clef_classification(
                    input[self.dimensions["clef"]],
                    target[self.dimensions["clef"]],
                )
                staff_idx_loss = self.staff_idx_classification(
                    input[self.dimensions["staff_idx"]],
                    target[self.dimensions["staff_idx"]],
                )
                return clef_loss + staff_idx_loss

        return Loss


class RestSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(
            in_dim,
            ["staff_idx", "voice_idx", "duration_type", "duration_dots"],
            TokenTensor.TOKEN_INDEXER[tk.Rest],
        )
        self.staff_idx_activation = nn.Softmax(dim=-1)
        self.voice_idx_activation = nn.Softmax(dim=-1)
        self.duration_type_activation = nn.Softmax(dim=-1)
        self.duration_dots_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Staff index -> Softmax
        x[self.dimensions["staff_idx"]] = self.staff_idx_activation(x[self.dimensions["staff_idx"]])
        # Voice index -> Softmax
        x[self.dimensions["voice_idx"]] = self.voice_idx_activation(x[self.dimensions["voice_idx"]])
        # Duration type -> Softmax
        x[self.dimensions["duration_type"]] = self.duration_type_activation(
            x[self.dimensions["duration_type"]]
        )
        # Duration dots -> Softmax
        x[self.dimensions["duration_dots"]] = self.duration_dots_activation(
            x[self.dimensions["duration_dots"]]
        )
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.staff_idx_classification = nn.NLLLoss()
                self.voice_idx_classification = nn.NLLLoss()
                self.duration_type_classification = nn.NLLLoss()
                self.duration_dots_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                staff_idx_loss = self.staff_idx_classification(
                    input[self.dimensions["staff_idx"]],
                    target[self.dimensions["staff_idx"]],
                )
                voice_idx_loss = self.voice_idx_classification(
                    input[self.dimensions["voice_idx"]],
                    target[self.dimensions["voice_idx"]],
                )
                duration_type_loss = self.duration_type_classification(
                    input[self.dimensions["duration_type"]],
                    target[self.dimensions["duration_type"]],
                )
                duration_dots_loss = self.duration_dots_classification(
                    input[self.dimensions["duration_dots"]],
                    target[self.dimensions["duration_dots"]],
                )
                return staff_idx_loss + voice_idx_loss + duration_type_loss + duration_dots_loss

        return Loss


class NoteSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(
            in_dim,
            [
                "staff_idx",
                "voice_idx",
                "duration_type",
                "duration_dots",
                "note_class",
                "note_octave",
                "note_accidental",
                "note_tied",
                "note_stem_direction",
            ],
            TokenTensor.TOKEN_INDEXER[tk.Note],
        )
        self.staff_idx_activation = nn.Softmax(dim=-1)
        self.voice_idx_activation = nn.Softmax(dim=-1)
        self.duration_type_activation = nn.Softmax(dim=-1)
        self.duration_dots_activation = nn.Softmax(dim=-1)
        self.note_class_activation = nn.Softmax(dim=-1)
        self.note_octave_activation = nn.Softmax(dim=-1)
        self.note_accidental_activation = nn.Softmax(dim=-1)
        self.note_tied_activation = nn.Sigmoid(dim=-1)
        self.note_stem_direction_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Staff index -> Softmax
        x[self.dimensions["staff_idx"]] = self.staff_idx_activation(x[self.dimensions["staff_idx"]])
        # Voice index -> Softmax
        x[self.dimensions["voice_idx"]] = self.voice_idx_activation(x[self.dimensions["voice_idx"]])
        # Duration type -> Softmax
        x[self.dimensions["duration_type"]] = self.duration_type_activation(
            x[self.dimensions["duration_type"]]
        )
        # Duration dots -> Softmax
        x[self.dimensions["duration_dots"]] = self.duration_dots_activation(
            x[self.dimensions["duration_dots"]]
        )
        # Note class -> Softmax
        x[self.dimensions["note_class"]] = self.note_class_activation(
            x[self.dimensions["note_class"]]
        )
        # Note octave -> Softmax
        x[self.dimensions["note_octave"]] = self.note_octave_activation(
            x[self.dimensions["note_octave"]]
        )
        # Note accidental -> Softmax
        x[self.dimensions["note_accidental"]] = self.note_accidental_activation(
            x[self.dimensions["note_accidental"]]
        )
        # Note tied -> Sigmoid
        x[self.dimensions["note_tied"]] = self.note_tied_activation(x[self.dimensions["note_tied"]])
        # Note stem direction -> Softmax
        x[self.dimensions["note_stem_direction"]] = self.note_stem_direction_activation(
            x[self.dimensions["note_stem_direction"]]
        )
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.staff_idx_classification = nn.NLLLoss()
                self.voice_idx_classification = nn.NLLLoss()
                self.duration_type_classification = nn.NLLLoss()
                self.duration_dots_classification = nn.NLLLoss()
                self.note_class_classification = nn.NLLLoss()
                self.note_octave_classification = nn.NLLLoss()
                self.note_accidental_classification = nn.NLLLoss()
                self.note_tied_classification = nn.BCELoss()
                self.note_stem_direction_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                staff_idx_loss = self.staff_idx_classification(
                    input[self.dimensions["staff_idx"]],
                    target[self.dimensions["staff_idx"]],
                )
                voice_idx_loss = self.voice_idx_classification(
                    input[self.dimensions["voice_idx"]],
                    target[self.dimensions["voice_idx"]],
                )
                duration_type_loss = self.duration_type_classification(
                    input[self.dimensions["duration_type"]],
                    target[self.dimensions["duration_type"]],
                )
                duration_dots_loss = self.duration_dots_classification(
                    input[self.dimensions["duration_dots"]],
                    target[self.dimensions["duration_dots"]],
                )
                note_class_loss = self.note_class_classification(
                    input[self.dimensions["note_class"]],
                    target[self.dimensions["note_class"]],
                )
                note_octave_loss = self.note_octave_classification(
                    input[self.dimensions["note_octave"]],
                    target[self.dimensions["note_octave"]],
                )
                note_accidental_loss = self.note_accidental_classification(
                    input[self.dimensions["note_accidental"]],
                    target[self.dimensions["note_accidental"]],
                )
                note_tied_loss = self.note_tied_classification(
                    input[self.dimensions["note_tied"]],
                    target[self.dimensions["note_tied"]],
                )
                note_stem_direction_loss = self.note_stem_direction_classification(
                    input[self.dimensions["note_stem_direction"]],
                    target[self.dimensions["note_stem_direction"]],
                )

                return (
                    staff_idx_loss
                    + voice_idx_loss
                    + duration_type_loss
                    + duration_dots_loss
                    + note_class_loss
                    + note_octave_loss
                    + note_accidental_loss
                    + note_tied_loss
                    + note_stem_direction_loss
                )

        return Loss


class ChordStartSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(
            in_dim,
            ["voice_idx", "duration_type", "duration_dots"],
            TokenTensor.TOKEN_INDEXER[tk.ChordStart],
        )
        self.voice_idx_activation = nn.Softmax(dim=-1)
        self.duration_type_activation = nn.Softmax(dim=-1)
        self.duration_dots_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Voice index -> Softmax
        x[self.dimensions["voice_idx"]] = self.voice_idx_activation(x[self.dimensions["voice_idx"]])
        # Duration type -> Softmax
        x[self.dimensions["duration_type"]] = self.duration_type_activation(
            x[self.dimensions["duration_type"]]
        )
        # Duration dots -> Softmax
        x[self.dimensions["duration_dots"]] = self.duration_dots_activation(
            x[self.dimensions["duration_dots"]]
        )
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.voice_idx_classification = nn.NLLLoss()
                self.duration_type_classification = nn.NLLLoss()
                self.duration_dots_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                voice_idx_loss = self.voice_idx_classification(
                    input[self.dimensions["voice_idx"]],
                    target[self.dimensions["voice_idx"]],
                )
                duration_type_loss = self.duration_type_classification(
                    input[self.dimensions["duration_type"]],
                    target[self.dimensions["duration_type"]],
                )
                duration_dots_loss = self.duration_dots_classification(
                    input[self.dimensions["duration_dots"]],
                    target[self.dimensions["duration_dots"]],
                )
                return voice_idx_loss + duration_type_loss + duration_dots_loss

        return Loss


class ChordEndSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(in_dim, ["voice_idx"], TokenTensor.TOKEN_INDEXER[tk.ChordEnd])
        self.voice_idx_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Voice index -> Softmax
        x[self.dimensions["voice_idx"]] = self.voice_idx_activation(x[self.dimensions["voice_idx"]])
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.voice_idx_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                voice_idx_loss = self.voice_idx_classification(
                    input[self.dimensions["voice_idx"]],
                    target[self.dimensions["voice_idx"]],
                )
                return voice_idx_loss

        return Loss


class TupletStartSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(
            in_dim,
            [
                "voice_idx",
                "tuplet_actual_notes",
                "tuplet_normal_notes",
                "duration_type",
                "duration_dots",
            ],
            TokenTensor.TOKEN_INDEXER[tk.TupletStart],
        )
        self.voice_idx_activation = nn.Softmax(dim=-1)
        self.duration_type_activation = nn.Softmax(dim=-1)
        self.duration_dots_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Voice index -> Softmax
        x[self.dimensions["voice_idx"]] = self.voice_idx_activation(x[self.dimensions["voice_idx"]])
        # Tuplet actual notes -> Nothing to do
        # Tuplet normal notes -> Nothing to do
        # Duration type -> Softmax
        x[self.dimensions["duration_type"]] = self.duration_type_activation(
            x[self.dimensions["duration_type"]]
        )
        # Duration dots -> Softmax
        x[self.dimensions["duration_dots"]] = self.duration_dots_activation(
            x[self.dimensions["duration_dots"]]
        )
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.voice_idx_classification = nn.NLLLoss()
                self.tuplet_actual_notes_regression = nn.MSELoss()
                self.tuplet_normal_notes_regression = nn.MSELoss()
                self.duration_type_classification = nn.NLLLoss()
                self.duration_dots_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                voice_idx_loss = self.voice_idx_classification(
                    input[self.dimensions["voice_idx"]],
                    target[self.dimensions["voice_idx"]],
                )
                tuplet_actual_notes_loss = self.tuplet_actual_notes_regression(
                    input[self.dimensions["tuplet_actual_notes"]],
                    target[self.dimensions["tuplet_actual_notes"]],
                )
                tuplet_normal_notes_loss = self.tuplet_normal_notes_regression(
                    input[self.dimensions["tuplet_normal_notes"]],
                    target[self.dimensions["tuplet_normal_notes"]],
                )
                duration_type_loss = self.duration_type_classification(
                    input[self.dimensions["duration_type"]],
                    target[self.dimensions["duration_type"]],
                )
                duration_dots_loss = self.duration_dots_classification(
                    input[self.dimensions["duration_dots"]],
                    target[self.dimensions["duration_dots"]],
                )
                return (
                    voice_idx_loss
                    + tuplet_actual_notes_loss
                    + tuplet_normal_notes_loss
                    + duration_type_loss
                    + duration_dots_loss
                )

        return Loss


class TupletEndSpecificPredictor(TokenSpecificPreditor):
    def __init__(self, in_dim: int):
        super().__init__(in_dim, ["voice_idx"], TokenTensor.TOKEN_INDEXER[tk.TupletEnd])
        self.voice_idx_activation = nn.Softmax(dim=-1)

    def predictor_specific_post_processing(self, x: torch.Tensor):
        # Voice index -> Softmax
        x[self.dimensions["voice_idx"]] = self.voice_idx_activation(x[self.dimensions["voice_idx"]])
        return x

    @classmethod
    def get_loss(cls):
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.voice_idx_classification = nn.NLLLoss()

            def forward(self, input: torch.Tensor, target: torch.Tensor):
                voice_idx_loss = self.voice_idx_classification(
                    input[self.dimensions["voice_idx"]],
                    target[self.dimensions["voice_idx"]],
                )
                return voice_idx_loss

        return Loss


if __name__ == '__main__':
    b = 5
    n = 550
    t = 700

    random_performance_streams = {
        "pitch": torch.randn(b, n, 128),
        "onset": torch.randn(b, n, 200),
        "duration": torch.randn(b, n, 200),
        "velocity": torch.randn(b, n, 8),
    }
    random_tgt = TokenTensor(torch.randn(b, t, TokenTensor.TOTAL_DIMENSIONS))

    model = TokensTranscriberModel()

    # Test the model
    output = model(
        src_streams=random_performance_streams,
        tgt_tokens=random_tgt,
    )
    print("Output shape:", output.shape)
