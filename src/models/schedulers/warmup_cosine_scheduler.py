"""
A learning rate scheduler with linear warmup and cosine decay.

Adapted from the code of Bayer's paper:
"End-to-end Piano Performance-MIDI to Score Conversion with Transformers"
"""

import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `total_steps - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        # progress after warmup
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    scheduler = WarmupCosineScheduler(
        optimizer=torch.optim.AdamW(params=[torch.nn.Parameter(torch.zeros(1))]),
        warmup_steps=1000,
        total_steps=10000,
        cycles=0.0,
    )

    lrs = []
    for step in range(10000):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plt.plot(lrs)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Warmup Cosine Learning Rate Scheduler")
    plt.show()
