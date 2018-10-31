# Jonas Rauber, 2018-09-30

from functools import wraps
import torch


def auto_batch(f, max_batch_size, n_args=1):
    """Calls f with batches of xs and concatenates the results.

    Supports multiple inputs and outputs."""

    assert max_batch_size >= 1
    assert n_args >= 1

    @wraps(f)
    def wrapper(*args, **kwargs):
        assert len(args) >= n_args

        # separate args into those that should be batched
        # and those that should not
        xs = args[:n_args]
        args = args[n_args:]

        n = xs[0].shape[0]

        y = []
        for start in range(0, n, max_batch_size):
            xb = [x[start:start + max_batch_size] for x in xs]
            yb = f(*xb, *args, **kwargs)
            y.append(yb)
        if isinstance(yb, tuple):
            return tuple(torch.cat(yi) for yi in zip(*y))
        else:
            return torch.cat(y)

    return wrapper
