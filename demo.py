import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=5000, help="initial labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds/tasks")
parser.add_argument('--dataset_name', type=str, default="CIFAR100", choices=["CIFAR100"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling",  # Use Entropy Sampling for improved selection
                    choices=["RandomSampling",
                             "LeastConfidence",
                             "MarginSampling",
                             "EntropySampling",
                             "LeastConfidenceDropout",
                             "MarginSamplingDropout",
                             "EntropySamplingDropout",
                             "KMeansSampling",
                             "KCenterGreedy",
                             "BALDDropout",
                             "AdversarialBIM",
                             "AdversarialDeepFool"], help="query strategy")
args = parser.parse_args()
pprint(vars(args))
print()

# Fix random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# Use CUDA if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load CIFAR-100 dataset and network
dataset = get_dataset(args.dataset_name)
net = get_net(args.dataset_name, device)
strategy = get_strategy(args.strategy_name)(dataset, net)

# Initialize experiment with initial labeled samples
dataset.initialize_labels(args.n_init_labeled)
print(f"Initial labeled pool: {args.n_init_labeled}")
print(f"Unlabeled pool: {dataset.n_pool - args.n_init_labeled}")
print(f"Testing pool: {dataset.n_test}")
print()

# Start training with active learning and continual task updates
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
print(f"Round 0 Testing Accuracy: {dataset.cal_test_acc(preds)}")

# Continual learning over rounds/tasks
for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")

    # Update dataset to include the next 10 classes
    dataset.update_task_classes()

    # Check remaining unlabeled samples
    remaining_unlabeled = dataset.n_pool - np.sum(dataset.labeled_idxs)
    if remaining_unlabeled < args.n_query:
        print(f"Only {remaining_unlabeled} unlabeled samples remaining, adjusting query size.")
        n_query = remaining_unlabeled
    else:
        n_query = args.n_query

    # Active learning query for new labeled samples
    query_idxs = strategy.query(n_query)
    strategy.update(query_idxs)

    # Retrain on new combined labeled data
    strategy.train()

    # Calculate accuracy on the complete test set
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round {rd} Testing Accuracy: {dataset.cal_test_acc(preds)}")

