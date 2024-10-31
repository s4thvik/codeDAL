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
        self.seen_classes = set()

        # Mapping from original class indices to seen class indices
        self.class_to_idx = {}
        self.idx_to_class = {}

    def initialize_labels(self, num):
        # Generate initial labeled pool with class balance across all classes
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        per_class_num = num // self.num_classes
        labeled_count = 0

        for cls in range(self.num_classes):
            cls_idxs = tmp_idxs[self.Y_train[tmp_idxs] == cls][:per_class_num]
            self.labeled_idxs[cls_idxs] = True
            labeled_count += len(cls_idxs)
            self.seen_classes.add(cls)

        # Fill remaining labels if any
        if labeled_count < num:
            remaining = num - labeled_count
            unlabeled_idxs = tmp_idxs[~self.labeled_idxs]
            extra_idxs = unlabeled_idxs[:remaining]
            self.labeled_idxs[extra_idxs] = True
            self.seen_classes.update(self.Y_train[extra_idxs])

        # Update class mappings
        self.update_class_mappings()

    def update_task_classes(self):
        """Advance to the next task without labeling additional samples."""
        start_class = self.task_counter * self.classes_per_task
        end_class = start_class + self.classes_per_task
        new_classes = list(range(start_class, min(end_class, self.num_classes)))
        self.current_classes.extend(new_classes)
        self.seen_classes.update(new_classes)

        # Update class mappings
        self.update_class_mappings()

        # Do not label any new samples here
        self.task_counter += 1

    def update_class_mappings(self):
        """Update mappings from original class indices to seen class indices."""
        seen_classes_sorted = sorted(self.seen_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(seen_classes_sorted)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(seen_classes_sorted)}

    def get_seen_classes(self):
        return sorted(list(self.seen_classes))

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        Y_labeled_mapped = np.array([self.class_to_idx[y] for y in self.Y_train[labeled_idxs]])
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], Y_labeled_mapped)

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        Y_unlabeled_mapped = np.array([self.class_to_idx.get(y, -1) for y in self.Y_train[unlabeled_idxs]])
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], Y_unlabeled_mapped)

    def get_test_data(self, seen_classes=None):
        if seen_classes is not None:
            mask = np.isin(self.Y_test, seen_classes)
            X_test_filtered = self.X_test[mask]
            Y_test_filtered = self.Y_test[mask]
            Y_test_mapped = np.array([self.class_to_idx[y] for y in Y_test_filtered])
            return self.handler(X_test_filtered, Y_test_mapped)
        else:
            return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds, seen_classes=None):
        if seen_classes is not None:
            mask = np.isin(self.Y_test, seen_classes)
            test_labels = self.Y_test[mask]
            test_labels_mapped = np.array([self.class_to_idx[y] for y in test_labels])
            correct = (test_labels_mapped == preds).sum().item()
            total = len(test_labels_mapped)
            return correct / total
        else:
            return (self.Y_test == preds).sum().item() / self.n_test

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

