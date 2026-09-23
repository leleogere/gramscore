from typing import Callable

import torch
from tqdm import trange

import models.automata.tokens as tk
from models.automata.automata import ConstrainedAutomata
from models.automata.vectorization_torch import TokenTensor, ProblemType, clip
from utils.generation import get_most_probable_valid_tokens
from utils.musicxml import ARTICULATIONS_AND_ORNAMENTS


if __name__ == '__main__':
    torch.manual_seed(5)
    batch = 5
    # print(router_distributions)
    # print(expert_outputs)

    automata = [ConstrainedAutomata(num_voices=8) for _ in range(batch)]

    print("Start")
    best_tokens_full = [[] for _ in range(batch)]
    best_scores_full = [[] for _ in range(batch)]
    for i in trange(100):
        router_distributions = torch.randn(batch, len(TokenTensor.TOKEN_INDEXER)).softmax(dim=-1)
        router_distributions[..., TokenTensor.TOKEN_INDEXER[tk.EndOfScore]] /= 10
        expert_outputs = {
            ttype.as_str(): {
                attr: (
                    torch.randn(batch, TokenTensor.DIMS[attr]).softmax(dim=-1)
                    if TokenTensor.PROBLEM_TYPES[attr] == ProblemType.CLASSIFICATION
                    else torch.rand(batch, 1) * 4
                )
                for attr in TokenTensor.TOKEN_TYPE_TO_DIMS[ttype]
                if attr != "token_type"
            }
            for ttype in tk.TOKEN_TYPES
        }
        best_tokens, best_probas = get_most_probable_valid_tokens(
            router_distributions, expert_outputs, [a.state for a in automata], max_depth=1000
        )
        for b in range(batch):
            if best_tokens[b] is not None:
                best_tokens_full[b].append(best_tokens[b])
                best_scores_full[b].append(best_probas[b])
                automata[b].transition(best_tokens[b])
    # print(best_tokens)
    # print(best_probas)

    for b in range(batch):
        print(f"Best tokens for {b}:")
        for token, proba in zip(best_tokens_full[b], best_scores_full[b]):
            print(f"  {proba:.4f}  {token}")
        print()
