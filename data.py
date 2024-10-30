import numpy as np
import torch
from torchvision import datasets, transforms

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, classes_per_task=10):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        self.current_classes = []  # Track current task's classes
        self.classes_per_task = classes_per_task
        self.task_counter = 0  # Track the current task index

    def initialize_labels(self, num):
        # Generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def update_task_classes(self):
        """Advance to the next task and label new classes."""
        start_class = self.task_counter * self.classes_per_task
        end_class = start_class + self.classes_per_task
        self.current_classes = list(range(start_class, end_class))

        # Label the new classes for this task
        new_labeled_idxs = np.isin(self.Y_train, self.current_classes)
        self.labeled_idxs |= new_labeled_idxs  # Update labeled indices
        self.task_counter += 1

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test


def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

def get_CIFAR100(handler, classes_per_task=10, num_tasks=10):
    # Load CIFAR-100 with transformations and split into continual tasks
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True,
                                   transform=transforms.ToTensor())
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True,
                                  transform=transforms.ToTensor())
    
    X_train, Y_train = data_train.data, torch.LongTensor(data_train.targets)
    X_test, Y_test = data_test.data, torch.LongTensor(data_test.targets)
    
    # Initialize the dataset with task-based class increments
    return Data(X_train, Y_train, X_test, Y_test, handler, classes_per_task=classes_per_task)

