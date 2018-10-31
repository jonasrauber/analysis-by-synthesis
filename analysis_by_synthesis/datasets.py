from os.path import join
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from .merging_sampler import MergingSampler


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
