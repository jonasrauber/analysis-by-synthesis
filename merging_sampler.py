# Jonas Rauber, 2018-09-29

from itertools import repeat, chain, islice
from torch.utils.data import Sampler


class MergingSampler(Sampler):
    r"""Interleaves samples from other samplers.

    Arguments:
        samplers (sequence): samplers to sample from
        num_per_sampler (string, optional): Specifies what to do if the samplers
            have different lengths: 'avg' | 'min' | 'max'. If the samplers all
            have the same length, this does not have an effect. Returns duplicate
            indices if 'avg' or 'max' and unequal lengths. Does not return all
            indices if 'avg' or 'min' and unequal lengths. Default: 'avg'
    """

    def __init__(self, samplers, num_per_sampler='avg'):
        self.samplers = samplers

        lengths = (len(sampler) for sampler in samplers)
        if num_per_sampler == 'avg':
            self.length = sum(lengths)
        elif num_per_sampler == 'min':
            self.length = min(lengths) * len(samplers)
        elif num_per_sampler == 'max':
            self.length = max(lengths) * len(samplers)
        else:
            raise ValueError(f'{num_per_sampler} is not a valid value for num_per_sampler')

    def __iter__(self):
        # create infinite iterators from samplers
        iterators = [chain.from_iterable(repeat(sampler)) for sampler in self.samplers]

        # zip the iterators and flatten the result
        indices = chain.from_iterable(zip(*iterators))

        # limit the itertor
        return islice(indices, self.length)

    def __len__(self):
        return self.length
