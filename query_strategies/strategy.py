# strategy.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        # Placeholder for query logic
        # Implement your active learning query strategy here
        # For example, Entropy Sampling, Least Confidence, etc.
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs is not None:
            self.dataset.labeled_idxs[neg_idxs] = False
        
        # Get class distribution of newly added samples
        new_classes = self.dataset.Y_train[pos_idxs]
        class_counts = np.bincount(new_classes, minlength=self.dataset.num_classes)
        
        # Create class distribution directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        dist_file = 'logs/class_distribution.txt'
        
        # Log class distribution immediately
        round_num = len(open(dist_file).readlines()) // 4 if os.path.exists(dist_file) else 0
        with open(dist_file, 'a') as f:
            f.write(f"Round {round_num} class distribution:\n")
            for class_idx, count in enumerate(class_counts):
                if count > 0:  # Only log classes that got new samples
                    f.write(f"Class {class_idx}: {count} samples\n")
            f.write("-" * 50 + "\n")

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        loss, metrics = self.net.train_model(labeled_data)
        return loss

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

