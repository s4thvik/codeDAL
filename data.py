import numpy as np
import torch
from torchvision import datasets, transforms

import numpy as np
from torchvision import datasets

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        self.num_classes = len(np.unique(Y_train))

    def initialize_labels(self, num):
        # Calculate samples per class for initial uniform sampling
        samples_per_class = num // self.num_classes
        remaining = num % self.num_classes
        
        labeled_indices = []
        for i in range(self.num_classes):
            # Get all indices for current class
            class_indices = np.where(self.Y_train == i)[0]
            # Calculate how many samples to take from this class
            n_samples = samples_per_class + (1 if i < remaining else 0)
            # Randomly select samples from this class
            if len(class_indices) >= n_samples:
                selected = np.random.choice(class_indices, n_samples, replace=False)
                labeled_indices.extend(selected)
        
        # Set the selected indices as labeled
        self.labeled_idxs[labeled_indices] = True
        
        # Log initial class distribution
        class_counts = np.bincount(self.Y_train[labeled_indices], minlength=self.num_classes)
        with open('round_class.txt', 'w') as f:
            np.savetxt(f, [class_counts], fmt='%d', delimiter=' ')

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        return np.mean(self.Y_test == preds)

    def get_training_batch(self, batch_size):
        # Get a random batch of labeled data for continuous training
        labeled_idxs = np.where(self.labeled_idxs)[0]
        selected_idxs = np.random.choice(labeled_idxs, batch_size, replace=False)
        return self.handler(self.X_train[selected_idxs], self.Y_train[selected_idxs])

# Utility functions for loading datasets
def get_CIFAR100(handler):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
    X_train, Y_train = data_train.data, np.array(data_train.targets)
    X_test, Y_test = data_test.data, np.array(data_test.targets)
    return Data(X_train, Y_train, X_test, Y_test, handler)


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


#hi3
