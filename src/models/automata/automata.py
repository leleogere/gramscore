import random

from models.automata.io import token_sequence_to_partitura, partitura_to_token_sequence
from models.automata.tokens import *


class ConstrainedAutomata:
    def __init__(
        self,
        num_voices: int = 1,
        initial_measure_number: int = 0,
        allowed_pitches: set[int] = None,
    ):
        self.state = TranscriptionState(num_voices=num_voices, allowed_pitches=allowed_pitches)

    def can_transition(self, token: Token) -> bool:
        return token.is_token_allowed(self.state)

    def transition(self, token: Token, inplace: bool = True):
        if not self.can_transition(token):
            raise ValueError(f"Cannot produce {token} in {self.state}")
        token.transition_fn(self.state, in_place=inplace)

    def allowed_tokens_types(self) -> list[Token]:
        return [token for token in TOKEN_TYPES if token.is_token_type_allowed(self.state)]

    def transition_sequence(self, tokens: list[Token]):
        for token in tokens:
            self.transition(token)

    def is_token_allowed(self, token: AnyToken) -> bool:
        return token.is_token_allowed(self.state)


if __name__ == '__main__':
    from pprint import pprint
    import partitura as pt
    import partitura.score
    from pathlib import Path
    from utils.ansi_colors import Colors

    # Parse a token sequence
    if False:
        automata = ConstrainedAutomata()
        print(automata.state)
        tokens = [
            StartOfScore(),
            Bar(1, 4),
            Note(Fraction(0.5), "C", 4, 0, False),
            Rest(Fraction(0.5)),
            # Note(Fraction(0), "D", 4, 0, False),
            EndOfScore(),
        ]
        for i, token in enumerate(tokens):
            print("  ", token)
            automata.transition(token)
            print(automata.state)

    # Sample a token sequence
    if False:
        random.seed(5)
        automata = ConstrainedAutomata(8)
        sequence = [StartOfScore()]
        print(automata.state)
        for i in range(100000):
            # Get allowed token types and sample one
            tokens = automata.allowed_tokens_types()
            assert tokens, f"No tokens allowed at {automata.state}"
            token_type = random.choice(tokens)
            print(f"  {Colors.GREEN}{token_type}{Colors.END}")
            token = token_type.sample_valid_token(automata.state)

            # Limit KS changes
            if isinstance(token, KeySignature) and random.random() < 0.9:
                continue

            # Limit TS changes
            if isinstance(token, Bar) and random.random() < 0.7:
                prev_bar = next(reversed([t for t in sequence if isinstance(t, Bar)]), None)
                if prev_bar is not None:
                    token = prev_bar

            # Limit EOS
            if isinstance(token, EndOfScore) and len(sequence) < 1000:
                continue

            # Limit Tuplets
            if isinstance(token, TupletStart) and random.random() < 0.9:
                continue

            # # Limit rare voices
            # if hasattr(token, "voice_idx"):
            #     if (
            #         (
            #             token.voice_idx == 1 % 4
            #             and not automata.state.voices[token.voice_idx].started
            #             and random.random() < 1
            #         )
            #         or (
            #             token.voice_idx == 2 % 4
            #             and not automata.state.voices[token.voice_idx].started
            #             and random.random() < 1
            #         )
            #         or (
            #             token.voice_idx == 3 % 4
            #             and not automata.state.voices[token.voice_idx].started
            #             and random.random() < 1
            #         )
            #     ):
            #         continue
            #         # TODO: Attempt at limiting rare voices but too long as random

            sequence.append(token)
            print(f"    {Colors.BLUE}{token}{Colors.END}")
            automata.transition(token)
            print(automata.state)
            if isinstance(token, EndOfScore):
                print("End of score!")
                break
        print()
        # pprint(sequence)

        score = token_sequence_to_partitura(sequence)

        print(score.pretty())
        pt.save_musicxml(score, "/tmp/test.musicxml")

    # Parse a MusicXML file
    if True:
        path = Path("/home/gerel/Téléchargements/Chopin__Trois_Valses.musicxml")
        # path = Path(
        #     "/home/gerel/Documents/datasets/asap-dataset/Chopin/Etudes_op_25/5/xml_score.musicxml"
        # )
        print(path.exists())
        score = pt.load_musicxml(path)
        assert len(score.parts) == 1
        score = score.parts[0]
        score = pt.score.unfold_part_maximal(score)

        tokens = partitura_to_token_sequence(score)
        # for token in tokens[:150]:
        #     print(token)

        automata = ConstrainedAutomata(8)
        bar_idx = 0
        for token in tokens:
            print(" > ", token)
            if isinstance(token, Bar):
                bar_idx += 1
                print(f"Bar {bar_idx}")
            automata.transition(token)
            print(automata.state)
            print()
            if isinstance(token, EndOfScore):
                break

        # Write back the score to the disk
        score = token_sequence_to_partitura(tokens)
        pt.save_musicxml(score, "/tmp/trois_valses.musicxml")
