import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)

        # Add a small epsilon to prevent log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        
        # Compute entropy
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Select samples with highest entropy
        _, idxs = entropy.sort(descending=True)
        return unlabeled_idxs[idxs[:n]]

