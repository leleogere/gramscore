import random

from src.models.automata.io import token_sequence_to_partitura
from src.models.automata.tokens import *


class ConstrainedAutomata:
    def __init__(self):
        self.state = TranscriptionState()

    def can_transition(self, token: Token) -> bool:
        return token.is_token_allowed(self.state)

    def transition(self, token: Token, inplace: bool = True):
        if not self.can_transition(token):
            raise ValueError(f"Cannot transition from {self.state} with {token}")
        token.transition_fn(self.state, in_place=inplace)

    def allowed_tokens_types(self) -> list[Token]:
        return [token for token in TOKEN_TYPES if token.is_token_type_allowed(self.state)]


if __name__ == '__main__':
    from pprint import pprint
    import partitura as pt
    import partitura.score
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
    if True:
        random.seed(4)
        automata = ConstrainedAutomata()
        sequence = []
        print(automata.state)
        for i in range(10000):
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
            if isinstance(token, EndOfScore) and len(sequence) < 10:
                continue

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
