import argparse


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
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers to load data (default: 4)')

    args = parser.parse_args()
    return args
