# demo.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from pprint import pprint
from utils import get_dataset, get_net, get_strategy, params
from torch.linalg import matrix_rank
from sklearn.metrics import classification_report
from nets import ContinualBackpropNet

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="Initial labeled samples")
parser.add_argument('--n_query', type=int, default=2000, help="Number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="Number of rounds")
parser.add_argument('--dataset_name', type=str, default="CIFAR100",
                    choices=["CIFAR100", "CIFAR10", "SVHN", "MNIST"], help="Dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling",
                    choices=["RandomSampling", "LeastConfidence", "MarginSampling", "EntropySampling",
                             "LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout",
                             "KMeansSampling", "KCenterGreedy", "BALDDropout", "AdversarialBIM",
                             "AdversarialDeepFool"], help="Query strategy")
args = parser.parse_args()
args_dict = vars(args)

pprint(args_dict)
print()

# Set random seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for CUDA availability
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset, network, and strategy
dataset = get_dataset(args.dataset_name)
net_cls = get_net(args.dataset_name, device)  # Pass device parameter
net = ContinualBackpropNet(
    net_cls=net_cls,
    params=params[args.dataset_name],  # Use params directly from utils
    device=device,
    replacement_rate=1e-4,
    maturity_threshold=500,
    decay_rate=0.9
)
strategy_class = get_strategy(args.strategy_name)
strategy = strategy_class(dataset, net)

# Initialize labeled data
dataset.initialize_labels(args.n_init_labeled)

# Function to calculate percentage of dead units
def calculate_dead_units(layer):
    if isinstance(layer, nn.Linear):
        # Calculate weight norms for each output unit
        weight_norms = torch.norm(layer.weight, dim=1)  # [out_features]
        
        # Consider bias if it exists
        if layer.bias is not None:
            bias_contribution = torch.abs(layer.bias)
            total_contribution = weight_norms + bias_contribution
        else:
            total_contribution = weight_norms
        
        # A unit is considered dead if its total contribution is near zero
        dead_mask = total_contribution < 1e-6
        return (dead_mask.sum().item() / layer.weight.size(0)) * 100
    return 0
# Function to calculate effective rank of a layer's weight matrix
def calculate_effective_rank(layer):
    if isinstance(layer, nn.Linear):
        # SVD-based effective rank calculation
        U, S, V = torch.svd(layer.weight.data)
        normalized_singular_values = S / S.sum()
        entropy = -(normalized_singular_values * torch.log(normalized_singular_values + 1e-10)).sum()
        return torch.exp(entropy).item()
    return 0

# Initialize log files
experiment_log = open("exp_output.log", "w")
summary_log = open("summary_output.txt", "w")

# Function to track class distribution
def track_class_distribution(dataset, query_idxs, round_num):
    new_classes = dataset.Y_train[query_idxs]
    class_counts = np.bincount(new_classes, minlength=dataset.num_classes)
    
    # Create distribution string
    dist_str = f"\nClass Distribution for Round {round_num}:\n"
    dist_str += "-" * 40 + "\n"
    for class_idx, count in enumerate(class_counts):
        if count > 0:
            dist_str += f"Class {class_idx:3d}: {count:4d} samples\n"
    dist_str += "-" * 40 + "\n"
    return dist_str, class_counts

# Function to log statistics
def log_statistics(round_num, loss, acc, per_class_acc, dead_units, avg_dead_units, 
                  avg_weight_magnitude, avg_effective_rank, new_labeled, total_labeled, class_dist_str=None):
    log_entry = f"Round {round_num}:\n"
    if loss is not None:
        log_entry += f"  Average Loss: {loss:.4f}\n"
    log_entry += f"  Testing Accuracy: {acc:.4f}\n"
    
    # Add class distribution if available
    if class_dist_str:
        log_entry += class_dist_str
    
    log_entry += f"  Per-Class Accuracy:\n"
    for cls, metrics in per_class_acc.items():
        if isinstance(metrics, dict):
            log_entry += f"    Class {cls}: Precision: {metrics['precision']:.4f}, "
            log_entry += f"Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}\n"
    
    log_entry += f"  Network Statistics:\n"
    log_entry += f"    Avg Weight Magnitude: {avg_weight_magnitude:.4f}\n"
    log_entry += f"    Percentage of Dead Units: {avg_dead_units:.2f}%\n"
    log_entry += f"    Effective Rank: {avg_effective_rank:.2f}\n"
    log_entry += f"  Sample Statistics:\n"
    log_entry += f"    New Labeled Samples: {new_labeled}\n"
    log_entry += f"    Total Labeled Samples: {total_labeled}\n\n"

    print(log_entry)
    experiment_log.write(log_entry)
    summary_log.write(log_entry)

# Round 0 training
print("Round 0")
loss_round_0 = strategy.train()
preds = strategy.predict(dataset.get_test_data())
round_0_accuracy = dataset.cal_test_acc(preds)
per_class_acc = classification_report(dataset.Y_test, preds, output_dict=True, zero_division=0)

# Calculate initial statistics
dead_units = sum(calculate_dead_units(layer) for layer in net.clf.model.children() if isinstance(layer, nn.Linear))
avg_dead_units = dead_units / len([layer for layer in net.clf.model.children() if isinstance(layer, nn.Linear)])
avg_weight_magnitude = sum(layer.weight.abs().mean().item() for layer in net.clf.model.children() if isinstance(layer, nn.Linear)) / len([layer for layer in net.clf.model.children() if isinstance(layer, nn.Linear)])
avg_effective_rank = sum(calculate_effective_rank(layer) for layer in net.clf.model.children() if isinstance(layer, nn.Linear)) / len([layer for layer in net.clf.model.children() if isinstance(layer, nn.Linear)])

new_labeled = args.n_init_labeled
total_labeled = args.n_init_labeled

# Track initial class distribution
dist_str, _ = track_class_distribution(dataset, np.where(dataset.labeled_idxs)[0], 0)

log_statistics(
    round_num=0,
    loss=loss_round_0,
    acc=round_0_accuracy,
    per_class_acc=per_class_acc,
    dead_units=dead_units,
    avg_dead_units=avg_dead_units,
    avg_weight_magnitude=avg_weight_magnitude,
    avg_effective_rank=avg_effective_rank,
    new_labeled=new_labeled,
    total_labeled=total_labeled,
    class_dist_str=dist_str
)

# Active learning rounds
for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")
    query_idxs = strategy.query(args.n_query)
    strategy.update(query_idxs)
    
    # Track class distribution before training
    dist_str, _ = track_class_distribution(dataset, query_idxs, rd)
    
    loss = strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    round_accuracy = dataset.cal_test_acc(preds)
    per_class_acc = classification_report(dataset.Y_test, preds, output_dict=True, zero_division=0)
    
    # Calculate statistics
    dead_units = sum(calculate_dead_units(layer) for layer in net.clf.model.children() if isinstance(layer, nn.Linear))
    avg_dead_units = dead_units / len([layer for layer in net.clf.model.children() if isinstance(layer, nn.Linear)])
    avg_weight_magnitude = sum(layer.weight.abs().mean().item() for layer in net.clf.model.children() if isinstance(layer, nn.Linear)) / len([layer for layer in net.clf.model.children() if isinstance(layer, nn.Linear)])
    avg_effective_rank = sum(calculate_effective_rank(layer) for layer in net.clf.model.children() if isinstance(layer, nn.Linear)) / len([layer for layer in net.clf.model.children() if isinstance(layer, nn.Linear)])

    new_labeled = len(query_idxs)
    total_labeled += new_labeled

    log_statistics(
        round_num=rd,
        loss=loss,
        acc=round_accuracy,
        per_class_acc=per_class_acc,
        dead_units=dead_units,
        avg_dead_units=avg_dead_units,
        avg_weight_magnitude=avg_weight_magnitude,
        avg_effective_rank=avg_effective_rank,
        new_labeled=new_labeled,
        total_labeled=total_labeled,
        class_dist_str=dist_str
    )

# Close log files
experiment_log.close()
summary_log.close()

print("Experiment complete. Check 'exp_output.log' for detailed logs and 'summary_output.txt' for summary.")












