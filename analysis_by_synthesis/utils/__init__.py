import torch

from .merging_sampler import MergingSampler  # noqa: F401
from .auto_batch import auto_batch  # noqa: F401


def count_correct(predictions, labels):
    assert predictions.dim() == 2
    assert labels.dim() == 1

    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == labels).sum().item()
    return correct
