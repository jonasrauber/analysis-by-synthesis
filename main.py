#!/usr/bin/env python3
from os.path import join
import torch
from torch import optim
import torchvision
from tensorboardX import SummaryWriter

from analysis_by_synthesis.datasets import get_dataset, get_dataset_loaders
from analysis_by_synthesis.inference import RobustInference
from analysis_by_synthesis.architecture import ABS
from analysis_by_synthesis.args import get_args
from analysis_by_synthesis.train import train
from analysis_by_synthesis.test import test
from analysis_by_synthesis.sample import sample


def main():
    assert not hasattr(torchvision.datasets.folder, 'find_classes'), 'torchvision master required'

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
