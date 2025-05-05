from collections.abc import Iterable

from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


class RepeatSampler(BatchSampler):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler: BatchSampler):
        self.sampler = sampler

    @property
    def drop_last(self):
        return self.sampler.drop_last

    @property
    def batch_size(self):
        return self.sampler.batch_size

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        if isinstance(self.sampler, Iterable):
            return len(self.sampler)
        else:
            raise TypeError("Sampler must be iterable")


class FastDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        breakpoint()
        object.__setattr__(self, 'batch_sampler', RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
