#!/usr/bin/env python3
import argparse
from os.path import join
import numpy as np
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

# local imports
from merging_sampler import MergingSampler
from auto_batch import auto_batch
from custom_kernel import samplewise_loss_function as pairwise_loss_function


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

def get_dataset(dataset, no_augmentation):
    if dataset == 'mnist':
        return get_mnist_dataset(no_augmentation)
    if dataset == 'cifar':
        return get_cifar_dataset(no_augmentation)
    if dataset == 'gtsrb':
        return get_gtsrb_dataset(no_augmentation, top10=True, grayscale=False)
    if dataset == 'grayscale_gtsrb':
        return get_gtsrb_dataset(no_augmentation, top10=True, grayscale=True)
    raise ValueError('unknown dataset')


def get_mnist_dataset(no_augmentation):
    def mnist_data_augmentation():
        # import Augmentor locally, to avoid unnecessary dependencies
        # if not training or not using data augmentation
        import Augmentor
        p = Augmentor.Pipeline()
        p.zoom(probability=0.1, min_factor=0.9, max_factor=1.1)
        p.shear(probability=0.1, max_shear_left=6, max_shear_right=6)
        p.skew(probability=0.1, magnitude=0.25)
        p.random_distortion(probability=0.1, grid_width=3, grid_height=3, magnitude=1)
        return transforms.Compose([p.torch_transform(), transforms.ToTensor()])

    transform = transforms.ToTensor() if no_augmentation else mnist_data_augmentation()
    train_set = datasets.MNIST('./data/mnist', train=True, transform=transform, download=True)
    test_set = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())
    return train_set, test_set


def get_cifar_dataset(no_augmentation):
    augmentation = [] if no_augmentation else [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    normalize = [
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    transform_train = transforms.Compose(augmentation + normalize)
    transform_test = transforms.Compose(normalize)

    train_set = datasets.CIFAR10('./data/cifar10', train=True, transform=transform_train, download=True)
    test_set = datasets.CIFAR10('./data/cifar10', train=False, transform=transform_test)
    return train_set, test_set


def get_gtsrb_dataset(no_augmentation, *, top10, grayscale, size=None):
    if size is None:
        size = 28 if grayscale else 32

    preprocess = [transforms.Grayscale(num_output_channels=1)] if grayscale else []

    train_resize = [transforms.Resize(size)] if no_augmentation else \
        [transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1))]
    test_resize = [transforms.Resize(size)]

    normalize = [
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]

    transform_train = transforms.Compose(preprocess + train_resize + normalize)
    transform_test = transforms.Compose(preprocess + test_resize + normalize)

    class FilteredImageFolder(datasets.ImageFolder):
        top10 = [2, 1, 13, 12, 38, 10, 4, 5, 25, 9]

        def _find_classes(self, dir):
            classes, _ = super()._find_classes(dir)
            # filter classes
            classes = [classes[i] for i in self.top10]
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return classes, class_to_idx

    Dataset = FilteredImageFolder if top10 else datasets.ImageFolder

    path = '/gpfs01/bethge/data/GTSRB'
    train_set = Dataset(join(path, 'train'), transform=transform_train)
    test_set = Dataset(join(path, 'val'), transform=transform_test)

    return train_set, test_set


def get_dataset_loaders(train_set, test_set, use_cuda, args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_kwargs = {'shuffle': True} if args.no_balancing \
        else {'sampler': create_balanced_sampler(train_set)}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, **train_kwargs, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


# ----------------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.conv_mu = nn.Conv2d(64, n_latents, 5)
        self.conv_logvar = nn.Conv2d(64, n_latents, 5)

    def forward(self, x):
        shared = self.shared(x)
        mu = self.conv_mu(shared)
        logvar = self.conv_logvar(shared)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_latents, 32, 4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 16, 5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class VAE(nn.Module):
    def __init__(self, n_latents):
        super().__init__()

        self.n_latents = n_latents
        self.encoder = Encoder(self.n_latents)
        self.decoder = Decoder(self.n_latents)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class ABS(nn.Module):
    """ABS model implementation that performs variational inference
    and can be used for training."""

    def __init__(self, n_classes, n_latents_per_class, beta):
        super().__init__()

        self.beta = beta
        self.vaes = nn.ModuleList([VAE(n_latents_per_class) for _ in range(n_classes)])

    def forward(self, x):
        outputs = [vae(x) for vae in self.vaes]
        recs, mus, logvars = zip(*outputs)
        recs, mus, logvars = torch.stack(recs), torch.stack(mus), torch.stack(logvars)
        losses = [samplewise_loss_function(x, *output, self.beta) for output in outputs]
        losses = torch.stack(losses)
        assert losses.dim() == 2
        logits = -losses.transpose(0, 1)
        return logits, recs, mus, logvars


# ----------------------------------------------------------------------------
# Robust inference
# ----------------------------------------------------------------------------

class RobustInference(nn.Module):
    """Takes a trained ABS model and replaces its variational inference
    with robust inference."""

    def __init__(self, abs_model, device, n_samples, n_iterations, *, fraction_to_dismiss, lr, radius, custom_kernel):
        super().__init__()

        self.abs = abs_model
        self.vaes = abs_model.vaes
        self.lr = lr
        self.beta = abs_model.beta
        self.radius = radius
        self.custom_kernel = custom_kernel
        self.name = f'{n_samples}_{n_iterations}'

        # create a set of random latents that we will reuse
        n_latents = self.vaes[0].n_latents
        self.z = self.draw_random_latents(n_samples, n_latents, fraction_to_dismiss).to(device)

        # assuming that z's were sampled from a normal distribution with mean = z, var = 1
        # note that we haven't acutally sampled z; instead z is simply mu
        self.mu = self.z
        self.logvar = torch.tensor(0.).to(device)

        self.cached_reconstructions = {}

        assert n_iterations >= 0, 'n_iterations must be non-negative'
        self.gradient_descent_iterations = n_iterations

    @staticmethod
    def draw_random_latents(n_samples, n_latents, fraction_to_dismiss):
        assert 0 <= fraction_to_dismiss < 1

        z = torch.randn(int(n_samples / (1 - fraction_to_dismiss)), n_latents, 1, 1)

        if z.size()[0] > n_samples:
            # ignore the least likely samples
            d = torch.sum(z ** 2, dim=(1, 2, 3))
            _, best = torch.sort(d)
            best = best[:n_samples]
            z = z[best]

        return z

    @staticmethod
    def clip_to_sphere_(z, radius):
        """Clips latents to a sphere. Operates in-place!

        This function assumes that the shape of z is
        (n_classes, batch_size, *latents_shape)"""

        if radius == np.inf:
            return

        # flatten the latent dimensions because torch.norm only works on one
        zr = z.reshape(z.size()[:2] + (-1,))

        length = torch.norm(zr, p=2, dim=2)

        # determine latents that are larger than desired
        mask = length > radius

        # add missing singleton dimensions to the end
        length = length.view(length.size() + (1,) * (z.dim() - length.dim()))

        z[mask] = z[mask] / length[mask] * radius

    def invalidate_cache(self):
        self.cached_reconstructions = {}

    def forward(self, x):
        """This performs robust inference by finding the optimal latents for
        each VAE using optimization rather than the encoder network."""

        if self.custom_kernel:
            samplewise_loss_function_b = pairwise_loss_function
        else:
            # calculating the loss for all pairs of inputs and reconstructions
            # can be extremely memory consuming; batch_size * n_samples * input_size
            # -> we therefore wrap our loss function with auto_batch

            # determine a good batch size of the inputs given the number of
            # reconstructions (e.g. 80 or 8000)
            def get_batch_size():
                GiB = 2**30
                memory = 4 * GiB
                n_recs = len(self.z)
                input_size = int(np.prod(x.shape[-3:]))
                bytes_per_float = 4
                batch_size = int(memory / bytes_per_float / input_size / n_recs)
                return batch_size

            samplewise_loss_function_b = auto_batch(samplewise_loss_function, get_batch_size(), n_args=1)

        with torch.no_grad():
            losses = []
            recs = []
            mus = []
            for vae in self.vaes:
                # pass the random latents through the VAEs
                if vae not in self.cached_reconstructions:
                    self.cached_reconstructions[vae] = vae.decoder(self.z)
                rec = self.cached_reconstructions[vae]

                # determine the best latents for each sample in x given this VAE
                # -> add a second batch dimension to x that will be broadcasted to the number of reconstructions
                # -> add a second batch dimension to rec that will be broadcasted to the number of inputs in x
                loss = samplewise_loss_function_b(x.unsqueeze(1), rec.unsqueeze(0), self.mu, self.logvar, self.beta)
                assert loss.dim() == 2
                # take min over samples in z
                loss, indices = loss.min(dim=1)

                losses.append(loss)
                recs.append(rec[indices])
                mus.append(self.mu[indices])

            mus = torch.stack(mus)

            if self.gradient_descent_iterations > 0:
                # for each sample and VAE, try to improve the best latents
                # further using gradient descent
                mus = self.gradient_descent(x, mus)

                # update losses and recs
                recs = [vae.decoder(mu) for vae, mu in zip(self.vaes, mus)]
                losses = [samplewise_loss_function(x, rec, mu, self.logvar, self.beta)
                          for rec, mu in zip(recs, mus)]

            recs = torch.stack(recs)
            losses = torch.stack(losses)

            logits = -losses.transpose(0, 1)
            logvars = torch.zeros_like(mus)
            return logits, recs, mus, logvars

    def gradient_descent(self, x, z):
        with torch.enable_grad():
            # create a completely independent copy of z
            z = torch.tensor(z, requires_grad=True).to(z.device)
            optimizer = optim.Adam([z], lr=self.lr)

            for j in range(self.gradient_descent_iterations):
                optimizer.zero_grad()

                loss = 0
                for vae, zi in zip(self.vaes, z):
                    rec = vae.decoder(zi)
                    loss += samplewise_loss_function(x, rec, zi, self.logvar, self.beta).sum()

                loss.backward()
                optimizer.step()

                # must operate on .data because PyTorch doesn't allow
                # in-place modifications of a leaf Variable itself
                self.clip_to_sphere_(z.data, radius=self.radius)
        return z


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Training and testing
# ----------------------------------------------------------------------------

def create_balanced_sampler(dataset):
    """Creates a sampler that balances classes by interleaving them."""
    try:
        labels = dataset.targets.numpy()
    except AttributeError:
        # dataset.targets is a list of ints (e.g. ImageFolder)
        labels = np.asarray(dataset.targets)
    classes = np.unique(labels)
    samplers = [SubsetRandomSampler(np.flatnonzero(labels == c)) for c in classes]
    return MergingSampler(samplers)


def count_correct(predictions, labels):
    assert predictions.dim() == 2
    assert labels.dim() == 1

    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == labels).sum().item()
    return correct


def train(model, args, device, train_loader, optimizer, epoch, writer=None):
    model.train()

    epoch_loss = 0
    epoch_correct = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # training
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits, recs, mus, logvars = model(data)
        loss = abs_loss_function(data, targets, recs, mus, logvars, args.beta)
        loss.backward()
        optimizer.step()

        # logging
        loss = loss.item()
        epoch_loss += loss
        normalized_loss = loss / len(data)
        correct = count_correct(logits, targets)
        accuracy = 100 * correct / len(data)
        epoch_correct += correct

        if writer is not None:
            step = (epoch - 1) * len(train_loader.sampler) + batch_idx * args.batch_size
            writer.add_scalar('train/loss', normalized_loss, step)
            writer.add_scalar('train/accuracy', accuracy, step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{:5d}/{} ({:2.0f}%)]\tLoss: {:3.4f} ({:2.0f}%)'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.sampler),
                100 * batch_idx / len(train_loader), normalized_loss, accuracy))

    epoch_loss /= len(train_loader.sampler)
    epoch_accuracy = epoch_correct / len(train_loader.sampler)
    if writer is not None:
        step = epoch * len(train_loader.sampler)
        writer.add_scalar('train/epoch-loss', epoch_loss, step)
        writer.add_scalar('train/epoch-accuracy', epoch_accuracy, step)
    print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy * 100:.3f}')


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


def sample(model, device, step, writer):
    if writer is None:
        return
    model.eval()
    n_latents = model.vaes[0].n_latents
    with torch.no_grad():
        zs = torch.randn(12, n_latents, 1, 1).to(device)
        samples = torch.cat([vae.decoder(zs).cpu() for vae in model.vaes])
        grid = make_grid(samples, nrow=12)
        writer.add_image(f'samples', grid, step)


def get_args():
    parser = argparse.ArgumentParser(description='Analysis by Synthesis Model')

    parser.add_argument('--test-only', action='store_true', default=False,
                        help='same as --initial-evaluation --epochs 0')

    # control loading and saving
    parser.add_argument('--logdir', default=None, type=str,
                        help='path to the TensorBoard log directory (default: None)')
    parser.add_argument('--load', default=None, type=str,
                        help='file from which the model should be loaded')
    parser.add_argument('--save', default=None, type=str,
                        help='file to which the model should be saved (in addition to logdir)')

    # control training
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='alpha',
                        help='learning rate for training')

    # control loss function
    parser.add_argument('--beta', type=float, default=1,
                        help='scaling factor for the KLD loss term')

    # control logging and evaluation
    parser.add_argument('--initial-evaluation', action='store_true', default=False,
                        help='perform an initial evaluation before training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs-full-evaluation', type=int, default=10, metavar='N',
                        help='how many epochs to wait before a full (expensive) evaluation')

    # control dataset
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='dataset to use, e.g. mnist, cifar, gtsrb (default:mnist)')
    parser.add_argument('--no-balancing', action='store_true', default=False,
                        help='disables class balancing in batches')
    parser.add_argument('--no-augmentation', action='store_true', default=False,
                        help='disables data augmentation')

    # control inference
    parser.add_argument('--inference-lr', type=float, default=5e-2,
                        help='learning rate for Adam during inference')
    parser.add_argument('--fraction-to-dismiss', type=float, default=0.1,
                        help='increases number of random samples and then ignores the least likely ones')
    parser.add_argument('--clip-to-sphere', type=float, default=5,
                        help='limit on the norm of the latents when doing gradient descent during inference')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # control performance
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-custom-kernel', action='store_true', default=False,
                        help='calculates the loss without the custom cuda kernel')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers to load data (default: 1)')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.test_only:
        args.initial_evaluation = True
        args.epochs = 0

    first_epoch = 0 if args.initial_evaluation else 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the train and test set
    train_set, test_set = get_dataset(args.dataset, args.no_augmentation)
    train_loader, test_loader = get_dataset_loaders(train_set, test_set, use_cuda, args)
    samples_per_epoch = len(train_loader.sampler)

    # create the ABS model
    model = ABS(n_classes=10, n_latents_per_class=8, beta=args.beta).to(device)
    model.eval()

    # load weights
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    # create wrappers that perform robust inference
    kwargs = {
        'fraction_to_dismiss': args.fraction_to_dismiss,
        'lr': args.inference_lr,
        'radius': args.clip_to_sphere,
        'custom_kernel': not args.no_custom_kernel,
    }
    robust_inference1 = RobustInference(model, device, n_samples=80, n_iterations=0, **kwargs)
    robust_inference2 = RobustInference(model, device, n_samples=8000, n_iterations=0, **kwargs)
    robust_inference3 = RobustInference(model, device, n_samples=8000, n_iterations=50, **kwargs)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # create writer for TensorBoard
    writer = SummaryWriter(args.logdir) if args.logdir is not None else None

    # main loop
    for epoch in range(first_epoch, args.epochs + 1):
        if epoch > 0:
            # train for one epoch
            train(model, args, device, train_loader, optimizer, epoch, writer=writer)

            # model changed, so make sure reconstructions are regenerated
            robust_inference1.invalidate_cache()
            robust_inference2.invalidate_cache()
            robust_inference3.invalidate_cache()

        step = epoch * samples_per_epoch

        # common params for calls to test
        params = (args, device, test_loader, step, writer)

        # some evaluations can happen after every epoch because they are cheap
        test(model, *params)
        test(robust_inference1, *params)
        test(robust_inference2, *params)

        # expensive evaluations happen from time to time and at the end
        if epoch % args.epochs_full_evaluation == 0 or epoch == args.epochs:
            test(robust_inference3, *params)

        sample(model, device, step, writer)

    # save the model
    if args.logdir is not None:
        path = join(args.logdir, 'model.pth')
        torch.save(model.state_dict(), path)
        print(f'model saved to {path}')
    if args.save is not None:
        torch.save(model.state_dict(), args.save)
        print(f'model saved to {args.save}')

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
