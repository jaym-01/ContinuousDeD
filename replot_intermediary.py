import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Point to the specific run folder you want to replot
run_dir = "runs/medgrid_cont_gen_2_seed10" # Change this to the run you want
data_path = os.path.join(run_dir, 'intermediary_plot_data.npz')

# 2. Load the data
try:
    data = np.load(data_path)
except FileNotFoundError:
    print(f"Could not find {data_path}. Did the training run complete?")
    exit()

#plots possible: data['confusion_matrix'], data['ground_truth_img'], data['cvar_d_grid']
cm = data['confusion_matrix']

# 3. Create your custom plot (Example: Bigger fonts on Confusion Matrix)
fig, ax = plt.subplots(figsize=(8, 6)) # Made figure slightly larger
im = ax.imshow(cm, cmap='Blues')

# Make text much larger
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=24, fontweight='bold')

# Customize axis labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Not Dead-End', 'Dead-End'], fontsize=14)
ax.set_yticklabels(['Not Dead-End', 'Dead-End'], fontsize=14)
ax.set_title('Confusion Matrix (Customized)', fontsize=18)

plt.tight_layout()
plt.savefig('custom_confusion_matrix.svg')
print("Custom plot saved to 'custom_confusion_matrix.svg'")
plt.show()