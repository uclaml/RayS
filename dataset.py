import random
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_mnist_test_data(test_batch_size=1):
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader


def load_cifar10_test_data(test_batch_size=1):
    # CIFAR10 Dataset
    test_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader


def load_imagenet_test_data(test_batch_size=1, folder='../val/'):
    val_dataset = dsets.ImageFolder(
        folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    rand_seed = 42

    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)

    return val_loader
