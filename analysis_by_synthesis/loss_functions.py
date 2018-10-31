import numpy as np
import torch


def samplewise_loss_function(x, rec_x, mu, logvar, beta):
    """This is the loss function used during inference to calculate the logits.

    This function must only operate on the last the dimensions of x and rec_x.
    There can be varying number of additional dimensions before them!
    """
    input_size = int(np.prod(x.shape[-3:]))
    # in-place because d can easily require huge amounts of memory;
    # using a custom cuda kernel, we could combine diff, square and sum
    # and avoid all the memory allocations
    d = rec_x - x
    d.pow_(2)
    L2squared = d.sum((-1, -2, -3)) / input_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=((-1, -2, -3))) / input_size
    # note that the KLD sum is over the latents, not over the input size
    return L2squared + beta * KLD


def vae_loss_function(x, rec_x, mu, logvar, beta):
    """Loss function to train a VAE summed over all elements and batch."""

    L2squared = torch.sum((rec_x - x).pow(2))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L2squared + beta * KLD


def abs_loss_function(x, labels, recs, mus, logvars, beta):
    """Loss function of the full ABS model

    Args:
        x (Tensor): batch of inputs
        labels (Tensor): batch of labels corresponding to the inputs
        recs (Tensor): reconstruction from each VAE (dim 0) for each sample (dim 1)
        mus (Tensor): mu from each VAE (dim 0) for each sample (dim 1)
        logvars (Tensor): logvar from each VAE (dim 0) for each sample (dim 1)
    """
    N = len(x)
    C = len(recs)

    assert labels.size() == (N,)
    assert recs.size()[:2] == (C, N)
    assert mus.size()[:2] == (C, N)
    assert logvars.size()[:2] == (C, N)

    assert labels.min().item() >= 0
    assert labels.max().item() < C

    loss = 0
    for c, rec, mu, logvar in zip(range(C), recs, mus, logvars):
        # train each VAE on samples from one class
        mask = (labels == c)
        if mask.sum().item() == 0:
            # batch does not contain samples for this VAE
            continue
        loss += vae_loss_function(x[mask], rec[mask], mu[mask], logvar[mask], beta) / N
    return loss
