import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=5000, help="initial labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds/tasks")
parser.add_argument('--dataset_name', type=str, default="CIFAR100", choices=["CIFAR100", "MNIST"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling",
                    choices=["RandomSampling", "LeastConfidence", "MarginSampling", "EntropySampling",
                             "LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout",
                             "KMeansSampling", "KCenterGreedy", "BALDDropout", "AdversarialBIM",
                             "AdversarialDeepFool"], help="query strategy")
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
torch.backends.cudnn.enabled = False

# Use CUDA if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load dataset and network
dataset = get_dataset(args.dataset_name)
net = get_net(args.dataset_name, device)
strategy = get_strategy(args.strategy_name)(dataset, net)

# Initialize labeled and unlabeled pools
dataset.initialize_labels(args.n_init_labeled)
initial_labeled = f"Initial labeled pool: {args.n_init_labeled}"
unlabeled_pool = f"Unlabeled pool: {dataset.n_pool - args.n_init_labeled}"
testing_pool = f"Testing pool: {dataset.n_test}"

print(initial_labeled)
print(unlabeled_pool)
print(testing_pool)
exp_log_file.write(f"{initial_labeled}\n{unlabeled_pool}\n{testing_pool}\n\n")
summary_log_file.write(f"{initial_labeled}\n{unlabeled_pool}\n{testing_pool}\n\n")

# Track sample counts for each round
current_labeled_count = args.n_init_labeled

# Round 0 training and logging
print("Round 0")
exp_log_file.write("Round 0\n")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
round_0_accuracy = dataset.cal_test_acc(preds)
round_0_log = f"Round 0 Testing Accuracy: {round_0_accuracy}"
print(round_0_log)
exp_log_file.write(f"{round_0_log}\n")
summary_log_file.write(f"{round_0_log}\n")

# Continual learning over rounds/tasks
for rd in range(1, args.n_round + 1):
    round_start = f"Round {rd}"
    print(round_start)
    exp_log_file.write(f"{round_start}\n")

    # Update dataset to include the next 10 classes
    dataset.update_task_classes()

    # Active learning query for new labeled samples
    query_idxs = strategy.query(args.n_query)
    strategy.update(query_idxs)

    # Calculate and log sample counts
    new_labeled_count = len(query_idxs)
    current_labeled_count += new_labeled_count
    current_unlabeled_count = dataset.n_pool - current_labeled_count

    # Log sample count details
    sample_log = (f"Round {rd} Sample Stats - Total Labeled: {current_labeled_count}, "
                  f"Newly Labeled: {new_labeled_count}, Unlabeled: {current_unlabeled_count}")
    print(sample_log)
    exp_log_file.write(f"{sample_log}\n")
    summary_log_file.write(f"{sample_log}\n")

    # Retrain on new combined labeled data
    strategy.train()

    # Calculate accuracy on the complete test set
    preds = strategy.predict(dataset.get_test_data())
    round_accuracy = dataset.cal_test_acc(preds)
    round_log = f"Round {rd} Testing Accuracy: {round_accuracy}"
    print(round_log)
    exp_log_file.write(f"{round_log}\n")
    summary_log_file.write(f"{round_log}\n")

exp_log_file.write("Experiment log complete.\n")
summary_log_file.write("Summary log complete.\n")

# Close log files
exp_log_file.close()
summary_log_file.close()

print("Experiment complete. Check 'exp_output.log' for detailed logs and 'summary_output.txt' for summary.")

