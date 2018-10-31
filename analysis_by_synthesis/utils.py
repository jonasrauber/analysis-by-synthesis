import torch


def count_correct(predictions, labels):
    assert predictions.dim() == 2
    assert labels.dim() == 1

    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == labels).sum().item()
    return correct
