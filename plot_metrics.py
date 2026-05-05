import json
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_combined_metrics():
    # Load the saved data
    try:
        metrics_cont = load_metrics('metrics_cont.json')
        metrics_disc = load_metrics('metrics_disc.json')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure both training scripts have finished running and saved their JSON files.")
        return

    # Set up the plotting grid (2x2 for the 4 metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_keys = ['precision', 'recall', 'f1', 'biou']
    labels = ['Precision', 'Recall', 'F1-Score', 'Boundary IoU']
    
    # Ensure both datasets evaluated the same dead-end values
    n_vals = sorted(list(set(metrics_cont['n_dead_ends'])))

    for idx, (metric, label) in enumerate(zip(metrics_keys, labels)):
        ax = axes[idx]
        
        # Calculate Continuous Mean and Std
        means_cont, stds_cont = [], []
        for n in n_vals:
            vals = [metrics_cont[metric][i] for i in range(len(metrics_cont['n_dead_ends'])) if metrics_cont['n_dead_ends'][i] == n]
            means_cont.append(np.nanmean(vals))
            stds_cont.append(np.nanstd(vals))
            
        # Calculate Discrete Mean and Std
        means_disc, stds_disc = [], []
        for n in n_vals:
            vals = [metrics_disc[metric][i] for i in range(len(metrics_disc['n_dead_ends'])) if metrics_disc['n_dead_ends'][i] == n]
            means_disc.append(np.nanmean(vals))
            stds_disc.append(np.nanstd(vals))
            
        # Plot both on the same axis
        ax.errorbar(n_vals, means_cont, yerr=stds_cont, marker='o', 
                    label='Continuous Model', capsize=4, linewidth=2)
        ax.errorbar(n_vals, means_disc, yerr=stds_disc, marker='s', 
                    label='Discrete Model', capsize=4, linewidth=2, linestyle='--')
        
        # Formatting
        ax.set_xlabel('Number of Dead-ends', fontsize=12)
        ax.set_ylabel(f'{label} Score', fontsize=12)
        ax.set_title(f'{label} vs Number of Dead-ends', fontsize=14)
        ax.set_xticks(n_vals)
        ax.legend()
        ax.grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig('combined_metrics_vs_dead_ends.svg', dpi=150)
    print("Combined plot saved successfully to 'combined_metrics_vs_dead_ends.svg'")
    plt.show()

if __name__ == '__main__':
    plot_combined_metrics()