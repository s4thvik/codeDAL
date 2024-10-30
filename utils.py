from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler, CIFAR100_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10, get_CIFAR100
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, CIFAR100_Net, ResNet18
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

params = {
    'MNIST': {
        'n_epoch': 10,
        'train_args': {'batch_size': 64, 'num_workers': 1},
        'test_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5}
    },
    'FashionMNIST': {
        'n_epoch': 10,
        'train_args': {'batch_size': 64, 'num_workers': 1},
        'test_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5}
    },
    'SVHN': {
        'n_epoch': 20,
        'train_args': {'batch_size': 64, 'num_workers': 1},
        'test_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5}
    },
    'CIFAR10': {
        'n_epoch': 20,
        'train_args': {'batch_size': 64, 'num_workers': 1},
        'test_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.05, 'momentum': 0.3}
    },
    'CIFAR100': {
        'n_epoch': 25,
        'train_args': {'batch_size': 64, 'num_workers': 2},
        'test_args': {'batch_size': 1000, 'num_workers': 2},
        'optimizer_args': {'lr': 0.05, 'momentum': 0.3}
    }
}

def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return MNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    elif name == 'CIFAR100':
        return CIFAR100_Handler
    else:
        raise NotImplementedError(f"Handler for dataset {name} not implemented.")

def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST(get_handler(name))
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(name))
    elif name == 'SVHN':
        return get_SVHN(get_handler(name))
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name))
    elif name == 'CIFAR100':
        # Use custom function to handle CIFAR-100 in a continual learning setup
        return get_CIFAR100(get_handler(name), classes_per_task=10, num_tasks=10)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")

def get_net(name, device):
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device)
    elif name == 'CIFAR100':
        return Net(ResNet18, params[name], device)
    else:
        raise NotImplementedError(f"Network for dataset {name} not implemented.")

def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError(f"Strategy {name} not implemented.")

