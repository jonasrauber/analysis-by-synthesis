# Jonas Rauber, 2018-10-30

import numpy as np
import cupy as cp
import torch
import torch.utils.dlpack
from torch.utils.dlpack import to_dlpack, from_dlpack


kernel = r'''
    extern "C" __global__
    void ssd(const float* x, const float* y, float* z) {{
        const unsigned int n_y = {};
        const unsigned int input_size = {};

        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        unsigned int x_ind = input_size * (tid / n_y);
        unsigned int y_ind = input_size * (tid % n_y);

        float s = 0;
        for (unsigned int j = 0; j < input_size; j += 1) {{
            float diff = x[x_ind + j] - y[y_ind + j];
            float squared = diff * diff;
            s += squared;
        }}
        z[tid] = s;
    }}
    '''


def samplewise_loss_function(x, rec_x, mu, logvar, beta):
    """This is the loss function used during inference to calculate the logits.

    This function must only operate on the last the dimensions of x and rec_x.
    There can be varying number of additional dimensions before them!
    """

    # for now, this custom kernel version of the samplewise_loss_function
    # only works for a particular case

    y = rec_x

    x = cp.fromDlpack(to_dlpack(x))
    y = cp.fromDlpack(to_dlpack(y))

    assert x.ndim == 5
    assert y.ndim == 5
    assert x.shape[1] == 1
    assert y.shape[0] == 1

    n_x = x.shape[0]
    n_y = y.shape[1]

    assert x.shape[2:] == y.shape[2:]

    input_size = int(np.prod(x.shape[-3:]))

    x = x.reshape((n_x, input_size))
    y = y.reshape((n_y, input_size))

    L2squared = cupy_pairwise_ssd(x, y)
    L2squared = from_dlpack(L2squared.toDlpack())
    L2squared = L2squared / input_size

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=((-1, -2, -3))) / input_size
    # note that the KLD sum is over the latents, not over the input size
    return L2squared + beta * KLD


def cupy_pairwise_ssd(x, y):
    assert x.ndim == 2
    assert y.ndim == 2

    # in my experiments this was fastest
    block = 10

    n_x = len(x)
    n_y = len(y)

    input_size = x.shape[-1]

    pairwise_ssd_kernel = cp.RawKernel(kernel.format(n_y, input_size), 'ssd')

    total = n_x * n_y
    assert total % block == 0
    grid = total // block

    z = cp.zeros((n_x, n_y), dtype=cp.float32)
    pairwise_ssd_kernel((grid,), (block,), (x, y, z))
    return z
