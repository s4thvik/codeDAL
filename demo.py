import argparse
import numpy as np
import torch
from pprint import pprint

# Assuming utils.py provides the following functions
from utils import get_dataset, get_net, get_strategy

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="Initial labeled samples")  # Increased to 10,000
parser.add_argument('--n_query', type=int, default=2000, help="Number of queries per round")      # Increased to 2,000
parser.add_argument('--n_round', type=int, default=10, help="Number of rounds/tasks")
parser.add_argument('--dataset_name', type=str, default="CIFAR100",
                    choices=["CIFAR100", "CIFAR10", "SVHN", "MNIST"], help="Dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling",
                    choices=["RandomSampling", "LeastConfidence", "MarginSampling", "EntropySampling",
                             "LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout",
                             "KMeansSampling", "KCenterGreedy", "BALDDropout", "AdversarialBIM",
                             "AdversarialDeepFool"], help="Query strategy")
args = parser.parse_args()
args_dict = vars(args)

# Display and log the arguments
pprint(args_dict)
print()

# Open log files
exp_log_file = open("exp_output.log", "w")
summary_log_file = open("summary_output.txt", "w")

# Log initial arguments to both files
exp_log_file.write("Experiment Arguments:\n")
summary_log_file.write("Experiment Arguments:\n")
for arg, value in args_dict.items():
    arg_str = f"{arg}: {value}\n"
    exp_log_file.write(arg_str)
    summary_log_file.write(arg_str)
exp_log_file.write("\n")
summary_log_file.write("\n")

# Fix random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use CUDA if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load dataset and network
dataset = get_dataset(args.dataset_name)
net = get_net(args.dataset_name, device)
strategy_class = get_strategy(args.strategy_name)
strategy = strategy_class(dataset, net)

# Initialize labeled and unlabeled pools
dataset.initialize_labels(args.n_init_labeled)
initial_labeled = f"Initial labeled pool: {np.sum(dataset.labeled_idxs)}"
unlabeled_pool = f"Unlabeled pool: {dataset.n_pool - np.sum(dataset.labeled_idxs)}"
testing_pool = f"Testing pool: {dataset.n_test}"

print(initial_labeled)
print(unlabeled_pool)
print(testing_pool)
exp_log_file.write(f"{initial_labeled}\n{unlabeled_pool}\n{testing_pool}\n\n")
summary_log_file.write(f"{initial_labeled}\n{unlabeled_pool}\n{testing_pool}\n\n")

# Track sample counts for each round
current_labeled_count = np.sum(dataset.labeled_idxs)

# Round 0 training and logging
print("Round 0")
exp_log_file.write("Round 0\n")

# Update the model with the initial seen classes
net.update_seen_classes([dataset.class_to_idx[cls] for cls in dataset.get_seen_classes()])

strategy.train()

# Calculate accuracy on seen classes
seen_classes = dataset.get_seen_classes()
test_data = dataset.get_test_data(seen_classes=seen_classes)
preds = strategy.predict(test_data)
preds = preds.numpy()
round_0_accuracy = dataset.cal_test_acc(preds, seen_classes=seen_classes)
round_0_log = f"Round 0 Testing Accuracy on Seen Classes: {round_0_accuracy:.4f}"
print(round_0_log)
exp_log_file.write(f"{round_0_log}\n")
summary_log_file.write(f"{round_0_log}\n")

# Initialize statistics tracking
accuracies = [round_0_accuracy]
per_class_accuracies = []
forgetting = []

# Compute per-class accuracy for round 0
from sklearn.metrics import confusion_matrix

test_labels = test_data.Y
num_classes = len(seen_classes)
cm = confusion_matrix(test_labels, preds, labels=range(num_classes))
with np.errstate(divide='ignore', invalid='ignore'):
    class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
    class_acc[cm.sum(axis=1) == 0] = 0
per_class_accuracies.append(class_acc)

# Continual learning over rounds/tasks
for rd in range(1, args.n_round + 1):
    round_start = f"Round {rd}"
    print(round_start)
    exp_log_file.write(f"{round_start}\n")

    # Update dataset to include the next set of classes (no new samples are labeled here)
    dataset.update_task_classes()

    # Update the model with the current seen classes
    net.update_seen_classes([dataset.class_to_idx[cls] for cls in dataset.get_seen_classes()])

    # Active learning query for new labeled samples
    query_idxs = strategy.query(args.n_query)
    strategy.update(query_idxs)

    # Verify total labeled samples
    current_labeled_count = np.sum(dataset.labeled_idxs)
    current_unlabeled_count = dataset.n_pool - current_labeled_count

    # Log sample count details
    sample_log = (f"Round {rd} Sample Stats - Total Labeled: {current_labeled_count}, "
                  f"Newly Labeled: {len(query_idxs)}, Unlabeled: {current_unlabeled_count}")
    print(sample_log)
    exp_log_file.write(f"{sample_log}\n")
    summary_log_file.write(f"{sample_log}\n")

    # Retrain on new combined labeled data
    strategy.train()

    # Calculate accuracy on seen classes
    seen_classes = dataset.get_seen_classes()
    test_data = dataset.get_test_data(seen_classes=seen_classes)
    preds = strategy.predict(test_data)
    preds = preds.numpy()
    round_accuracy = dataset.cal_test_acc(preds, seen_classes=seen_classes)
    round_log = f"Round {rd} Testing Accuracy on Seen Classes: {round_accuracy:.4f}"
    print(round_log)
    exp_log_file.write(f"{round_log}\n")
    summary_log_file.write(f"{round_log}\n")

    accuracies.append(round_accuracy)

    # Compute per-class accuracy
    num_classes = len(seen_classes)
    cm = confusion_matrix(test_data.Y, preds, labels=range(num_classes))
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        class_acc[cm.sum(axis=1) == 0] = 0
    per_class_accuracies.append(class_acc)

    # Compute forgetting
    if rd > 1:
        prev_class_acc = per_class_accuracies[-2]
        current_class_acc = class_acc
        forgetting_per_class = prev_class_acc - current_class_acc
        forgetting.append(forgetting_per_class)
        # Log forgetting
        forgetting_log = f"Round {rd} Forgetting per class: {forgetting_per_class}"
        print(forgetting_log)
        exp_log_file.write(f"{forgetting_log}\n")
        summary_log_file.write(f"{forgetting_log}\n")

exp_log_file.write("Experiment log complete.\n")
summary_log_file.write("Summary log complete.\n")

# Close log files
exp_log_file.close()
summary_log_file.close()

print("Experiment complete. Check 'exp_output.log' for detailed logs and 'summary_output.txt' for summary.")

