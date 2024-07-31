import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["SimSiam (baseline)", "Sim. + HVS (4 views)", "Sim. + HVS (8 views)"]
lin_100 = [68.2, 68.98, 68.93]
lin_200 = [69.76, 70.47, 70.33]

# Set up subplots with shared y-axis
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Reduce bar width
bar_width = 0.5

# Subplot for 100 epochs
axes[0].bar(methods, lin_100, color='darkcyan', alpha=0.7, width=bar_width)
axes[0].set_title('100 Epochs')
axes[0].set_ylabel('ImageNet Linear Evaluation Accuracy')
axes[0].set_ylim(67, 71)  # Set the y-axis range

# Subplot for 200 epochs
axes[1].bar(methods, lin_200, color='coral', alpha=0.7, width=bar_width)
axes[1].set_title('200 Epochs')
axes[1].set_ylim(67, 71)  # Set the y-axis range

# Reduce the font size of x-axis labels
for ax in axes:
    ax.tick_params(axis='x', labelsize=11)  # Adjust the label size as needed

fig.suptitle('Effect of number of views on linear evaluation')

# Adjust layout
plt.tight_layout()

plt.savefig('ncrops_plot.png')

# Show the plot
plt.show()
