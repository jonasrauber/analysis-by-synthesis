import tqdm
import torch
from torchvision.utils import make_grid

from .loss_functions import abs_loss_function
from .utils import count_correct


def test(model, args, device, test_loader, step, writer=None, max_batches=None):
    model.eval()
    suffix = '-' + model.name if hasattr(model, 'name') else ''

    N = len(test_loader.dataset)

    loss = 0
    correct = 0

    with torch.no_grad():
        # using a context manager for tqdm prevents messed up console outputs
        with tqdm.tqdm(test_loader) as t:
            for i, (data, targets) in enumerate(t):
                data = data.to(device)
                targets = targets.to(device)
                logits, recs, mus, logvars = model(data)
                loss += abs_loss_function(data, targets, recs, mus, logvars, args.beta).item()
                correct += count_correct(logits, targets)

                if i == 0 and writer is not None:
                    # up to 8 samples
                    n = min(data.size(0), 8)
                    # flatten VAE and batch dim into a single dim
                    shape = (-1,) + recs.size()[2:]
                    grid = torch.cat([data[:n], recs[:, :n].reshape(shape)])
                    grid = make_grid(grid, nrow=n)
                    writer.add_image(f'test/reconstructions{suffix}', grid, step)

                if i == max_batches:
                    # limit testing to a subset by passing max_batches
                    N = i * args.test_batch_size + len(data)
                    break

    loss /= N
    accuracy = 100 * correct / N
    print(f'====> Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{N} ({accuracy:.0f}%)\n')

    if writer is not None:
        writer.add_scalar(f'test/loss{suffix}', loss, step)
        writer.add_scalar(f'test/accuracy{suffix}', accuracy, step)
