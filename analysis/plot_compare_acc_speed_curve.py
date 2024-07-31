import matplotlib.pyplot as plt
import json
import os

# Define the paths to the directories containing eval.log files
experiment1_dir = '/work/dlclarge1/ferreira-simsiam/minsim_experiments/simsiam-minsim-transfer-learning-seed0'
# experiment1_dir = '/work/dlclarge1/ferreira-simsiam/minsim_experiments/1_completed/dino-minsim-baseline-vit_small_p16-ImageNet-ep100-bs512-select_cross-ncrops4-lr0.0005-wd0.04-out_dim65k-seed0'
# experiment2_dir = '/work/dlclarge1/ferreira-simsiam/minsim_experiments/simsiam-vanilla-transfer-learning-seed0'
experiment2_dir = '/work/dlclarge1/ferreira-simsiam/minsim_experiments/simsiam-vanilla-transfer-learning-seed0'
plot_name = "acc_speed_comparison_dino_100ep.png"
# plot_name = "acc_speed_comparison_simsiam_300ep.png"

# Extract experiment names from directory names
experiment1_name = os.path.basename(experiment1_dir)
experiment2_name = os.path.basename(experiment2_dir)

# Define the paths to the eval.log files of the two experiments
experiment1_path = os.path.join(experiment1_dir, 'eval.log')
experiment2_path = os.path.join(experiment2_dir, 'eval.log')

# Initialize dictionaries to store the first "train_acc1" value for each epoch in each experiment
train_acc1_experiment1 = {}
train_acc1_experiment2 = {}

# Read and extract data from experiment 1
with open(experiment1_path, 'r') as file1:
    for line in file1:
        data = json.loads(line)
        epoch = data['epoch']
        if epoch not in train_acc1_experiment1:
            train_acc1_experiment1[epoch] = data['train_acc1']

# Read and extract data from experiment 2
with open(experiment2_path, 'r') as file2:
    for line in file2:
        data = json.loads(line)
        epoch = data['epoch']
        if epoch not in train_acc1_experiment2:
            train_acc1_experiment2[epoch] = data['train_acc1']

experiment1_name = experiment1_name.replace("minsim", "hvs")

# Extract unique epochs
epochs_experiment1 = list(train_acc1_experiment1.keys())
epochs_experiment2 = list(train_acc1_experiment2.keys())

# Sort the epochs
epochs_experiment1.sort()
epochs_experiment2.sort()

# Extract the train_acc1 values for each epoch
train_acc1_experiment1_values = [train_acc1_experiment1[epoch] for epoch in epochs_experiment1]
train_acc1_experiment2_values = [train_acc1_experiment2[epoch] for epoch in epochs_experiment2]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_experiment1, train_acc1_experiment1_values, label=f'hvs')
plt.plot(epochs_experiment2, train_acc1_experiment2_values, label=f'vanilla')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title(f'{experiment1_name} vs {experiment2_name}')
plt.legend()
plt.grid(True)

plt.savefig(plot_name)

# Show the plot
plt.show()
