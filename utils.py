# utils.py
from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler, CIFAR100_Handler
from data import get_MNIST, get_SVHN, get_CIFAR10, get_CIFAR100
from nets import ContinualBackpropNet, MNIST_Net, SVHN_Net, CIFAR10_Net, CustomResNet18

# Import the active learning strategies from the query_strategies folder
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

params = {
    'CIFAR100': {
        'n_epoch': 100,  # Increased from 10 to 100 for better training
        'train_args': {
            'batch_size': 128,
            'num_workers': 2,
            'shuffle': True
        },
        'test_args': {
            'batch_size': 100,
            'num_workers': 2,
            'shuffle': False
        },
        'optimizer_args': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        'num_classes': 100
    }
}

def get_handler(name):
    if name == 'MNIST':
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
    elif name == 'SVHN':
        return get_SVHN(get_handler(name))
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name))
    elif name == 'CIFAR100':
        return get_CIFAR100(get_handler(name))
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")

def get_net(name, device):
    if name == 'MNIST':
        return MNIST_Net
    elif name == 'SVHN':
        return SVHN_Net
    elif name == 'CIFAR10':
        return CIFAR10_Net
    elif name == 'CIFAR100':
        return CustomResNet18
    else:
        raise NotImplementedError(f"Network for dataset {name} not implemented.")

def get_strategy(name):
    strategies = {
        "RandomSampling": RandomSampling,
        "LeastConfidence": LeastConfidence,
        "MarginSampling": MarginSampling,
        "EntropySampling": EntropySampling,
        "LeastConfidenceDropout": LeastConfidenceDropout,
        "MarginSamplingDropout": MarginSamplingDropout,
        "EntropySamplingDropout": EntropySamplingDropout,
        "KMeansSampling": KMeansSampling,
        "KCenterGreedy": KCenterGreedy,
        "BALDDropout": BALDDropout,
        "AdversarialBIM": AdversarialBIM,
        "AdversarialDeepFool": AdversarialDeepFool
    }
    if name in strategies:
        return strategies[name]
    else:
        raise NotImplementedError(f"Strategy {name} not implemented.")




