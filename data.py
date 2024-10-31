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

        self.num_classes = len(np.unique(Y_train))

    def initialize_labels(self, num):
        # Generate initial labeled pool with class balance
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        per_class_num = num // self.classes_per_task
        labeled_count = 0

        for cls in range(self.classes_per_task):
            cls_idxs = tmp_idxs[self.Y_train[tmp_idxs] == cls][:per_class_num]
            self.labeled_idxs[cls_idxs] = True
            labeled_count += len(cls_idxs)

        # Fill remaining labels if any
        if labeled_count < num:
            remaining = num - labeled_count
            unlabeled_idxs = tmp_idxs[~self.labeled_idxs[tmp_idxs]]
            extra_idxs = unlabeled_idxs[:remaining]
            self.labeled_idxs[extra_idxs] = True

    def update_task_classes(self):
        """Advance to the next task and label new classes."""
        start_class = self.task_counter * self.classes_per_task
        end_class = start_class + self.classes_per_task
        self.current_classes = list(range(start_class, min(end_class, self.num_classes)))

        # Label a subset of samples from the new classes
        n_per_class = 100  # Adjust as needed
        new_labeled_idxs = []
        for cls in self.current_classes:
            cls_idxs = np.where((self.Y_train == cls) & (~self.labeled_idxs))[0]
            np.random.shuffle(cls_idxs)
            cls_idxs = cls_idxs[:n_per_class]
            new_labeled_idxs.extend(cls_idxs)
        self.labeled_idxs[new_labeled_idxs] = True

        self.task_counter += 1

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data.numpy(), raw_train.targets.numpy(),
                raw_test.data.numpy(), raw_test.targets.numpy(), handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data, data_train.labels,
                data_test.data, data_test.labels, handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data, np.array(data_train.targets),
                data_test.data, np.array(data_test.targets), handler)

def get_CIFAR100(handler, classes_per_task=10):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)

    X_train, Y_train = data_train.data, np.array(data_train.targets)
    X_test, Y_test = data_test.data, np.array(data_test.targets)

    return Data(X_train, Y_train, X_test, Y_test, handler, classes_per_task=classes_per_task)

