from src.models.automata.tokens import *


class ConstrainedAutomata:
    def __init__(self):
        self.state = TranscriptionState()

    def can_transition(self, token: Token) -> bool:
        return token.is_token_allowed(self.state)

    def transition(self, token: Token):
        if not self.can_transition(token):
            raise ValueError(f"Cannot transition from {self.state} with {token}")
        token.transition_fn(self.state, in_place=True)

    def allowed_tokens(self):
        token_types = TOKEN_TYPES
        allowed = []
        for token_type in token_types:
            kwargs = token_type.allowed_kwargs(self.state)
            if all(value == None or len(value) > 0 for value in kwargs.values()):
                allowed.append(token_type)
        return allowed


if __name__ == '__main__':

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
        automata = ConstrainedAutomata()
        automata.transition(StartOfScore())
        allowed_tokens = automata.allowed_tokens()
        print(allowed_tokens)
