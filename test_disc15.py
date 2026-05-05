import sys
import os
import io
import pickle
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import gymnasium as gym
import json

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'toy_domain'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MedGrid'))

# Now import the continuous agent module so pickle knows about it
import agent_continuous

# Ensure your custom environment is imported
import MedGridGeneral.med_grid_general_env

# ==========================================
# CONFIGURATION
# ==========================================
SEEDS = [10, 20, 30, 40, 50]        # 5 seeds for statistical robustness
NUM_DEAD_END = [2,4,6,8,10]        # Evaluate for the number of dead-ends in the list
FRAMES = 100000             # Frames per training run
N_BINS = 15                 # Matches your notebook's bin setup
ACTION_SIZE = N_BINS * N_BINS
BINS = np.linspace(0.0, 10.0, N_BINS, dtype=np.float32)
ACTION_MAP = np.array([(x, y) for x in BINS for y in BINS], dtype=np.float32)
SIZE = 10.0
N_TAU = 32
ALPHA = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
FIXED_STATE = [10.0, 0.0]   # The state evaluated for metrics
DELTA_D = -0.5
DELTA_R = 0.5
N_CONT_SAMPLES = 100000
BIOU_RES = 200

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
class CpuUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA-backed torch storages onto CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'), weights_only=False)
        return super().find_class(module, name)

def load_pickle_cpu(path):
    with open(path, 'rb') as f:
        return CpuUnpickler(f).load()

def compute_action_cvar_discrete(network, state, n_tau, eta, device):
    n_keep = max(1, int(eta * n_tau))
    network.to(device)
    network.device = device
    if hasattr(network, 'pis'):
        network.pis = network.pis.to(device)
    network.eval()
    s_t = torch.tensor([state], dtype=torch.float32).to(device)
    with torch.no_grad():
        quantiles, _ = network.forward(s_t, num_tau=n_tau)
    q = quantiles[0].cpu().numpy()
    q_sorted = np.sort(q, axis=0)
    cvar = q_sorted[:n_keep, :].mean(axis=0)
    return cvar

def compute_action_cvar_grid_discrete(network, state, n_tau, eta, device):
    cvar = compute_action_cvar_discrete(network, state, n_tau, eta, device)
    return cvar.reshape(N_BINS, N_BINS), BINS

def nearest_discrete_indices(actions, action_bins):
    ix = np.abs(actions[:, 0][:, None] - action_bins[None, :]).argmin(axis=1)
    iy = np.abs(actions[:, 1][:, None] - action_bins[None, :]).argmin(axis=1)
    return ix, iy

def boundary_mask(binary_grid):
    g = binary_grid.astype(np.int8)
    shifted = (
        np.pad(g, ((1, 0), (0, 0)), mode='edge')[:-1, :]
        + np.pad(g, ((0, 1), (0, 0)), mode='edge')[1:, :]
        + np.pad(g, ((0, 0), (1, 0)), mode='edge')[:, :-1]
        + np.pad(g, ((0, 0), (0, 1)), mode='edge')[:, 1:]
    )
    return binary_grid & (shifted < 4)

def boundary_iou(gt_grid, pred_grid):
    gt_boundary = boundary_mask(gt_grid)
    pred_boundary = boundary_mask(pred_grid)
    intersection = (gt_boundary & pred_boundary).sum()
    union = (gt_boundary | pred_boundary).sum()
    return float(intersection) / float(union) if union > 0 else float('nan')


# ==========================================
# MAIN EXPERIMENT LOOP
# ==========================================
def run_experiment():
    metrics = {'n_dead_ends': [], 'seed': [], 'precision': [], 'recall': [], 'f1': [], 'biou': []}
    
    if isinstance(NUM_DEAD_END, int):
        dead_ends_to_test = np.arange(1, NUM_DEAD_END+1)
    else:
        dead_ends_to_test = NUM_DEAD_END

    for n in dead_ends_to_test:
        for seed in SEEDS:
            run_info = f"medgrid_gen_{n}_seed{seed}"
            run_dir = os.path.join("runs", run_info)
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"\n=======================================================")
            print(f"Training agent for {n} dead-ends, Seed {seed}")
            print(f"=======================================================")
            cmd = [
                "python", "toy_domain/run.py",
                "-env", "MedGridGeneral",
                "-action_mode", "discrete",
                "-n_bins", str(N_BINS),
                "-agent", "iqn",
                "-ded",
                "-frames", str(FRAMES),
                "-info", run_info,
                "-num_dead_ends", str(n),
                "-seed", str(seed)
            ]
            
            # Subprocess runs the command line dynamically
            subprocess.run(cmd, check=True)
            
            print(f"Evaluating agent for {n} dead-ends, Seed {seed}...")
            qd = load_pickle_cpu(os.path.join(run_dir, f"{run_info}_Qd.pkl")).qnetwork_local
            qr = load_pickle_cpu(os.path.join(run_dir, f"{run_info}_Qr.pkl")).qnetwork_local
            
            # Recreate env to retrieve procedural danger zones mapped by the seed
            env = MedGridGeneral.med_grid_general_env.MedGridGeneralEnv(num_dead_ends=n, scale=1.0, seed=seed)
            
            # 1. GROUND TRUTH MAP
            RESOLUTION = 400
            xs = np.linspace(0, SIZE, RESOLUTION)
            ys = np.linspace(0, SIZE, RESOLUTION)
            img = np.zeros((RESOLUTION, RESOLUTION, 4), dtype=np.uint8)
            colors = {
                'death': [255, 0, 0, 200],         # Red
                'dead_end': [255, 255, 0, 200],    # Yellow
                'recovery': [0, 100, 255, 200],    # Blue
                'neutral': [255, 255, 255, 255]    # White
            }
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    status, _ = env._check_collision(np.array([x, y]), env.danger_zones)
                    img[j, i] = colors[status]
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, origin='lower', extent=[0, SIZE, 0, SIZE], aspect='equal')
            ax.set_title(f'Ground Truth - {n} Dead-ends (Seed {seed})')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'ground_truth.svg'))
            plt.close()
            
            # 2. CVaR Q-VALUE HEATMAPS
            cvar_d_grid, action_bins = compute_action_cvar_grid_discrete(qd, FIXED_STATE, N_TAU, ALPHA, DEVICE)
            cvar_r_grid, _ = compute_action_cvar_grid_discrete(qr, FIXED_STATE, N_TAU, ALPHA, DEVICE)
            pred_action_grid = (cvar_d_grid <= DELTA_D) & (cvar_r_grid <= DELTA_R)
            
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            #Find the absolute max across both grids to ensure the color scale is identical
            max_abs_val = max(np.abs(cvar_d_grid).max(), np.abs(cvar_r_grid).max(), 1e-6)
            shared_norm = TwoSlopeNorm(vmin=-max_abs_val, vcenter=0, vmax=max_abs_val)
            
            im_d = axes[0].imshow(cvar_d_grid.T, origin='lower', extent=[0, 10, 0, 10], aspect='equal', cmap='RdBu', norm=shared_norm)
            axes[0].set_title(f'Q_d CVaR (alpha={ALPHA})')
            
            im_r = axes[1].imshow(cvar_r_grid.T, origin='lower', extent=[0, 10, 0, 10], aspect='equal', cmap='RdBu', norm=shared_norm)
            axes[1].set_title(f'Q_r CVaR (alpha={ALPHA})')
            
            # Add one colorbar that serves for both plots
            fig.colorbar(im_d, ax= axes[1], fraction=0.046, pad=0.04)


            """
            #Old logic with 2 colorbars
            norm_d = TwoSlopeNorm(vmin=-max(np.abs(cvar_d_grid).max(), 1e-6), vcenter=0, vmax=max(np.abs(cvar_d_grid).max(), 1e-6))
            im_d = axes[0].imshow(cvar_d_grid.T, origin='lower', extent=[0, 10, 0, 10], aspect='equal', cmap='RdBu', norm=norm_d)
            fig.colorbar(im_d, ax=axes[0])
            axes[0].set_title(f'Q_d CVaR (alpha={ALPHA})')
            
            norm_r = TwoSlopeNorm(vmin=-max(np.abs(cvar_r_grid).max(), 1e-6), vcenter=0, vmax=max(np.abs(cvar_r_grid).max(), 1e-6))
            im_r = axes[1].imshow(cvar_r_grid.T, origin='lower', extent=[0, 10, 0, 10], aspect='equal', cmap='RdBu', norm=norm_r)
            fig.colorbar(im_r, ax=axes[1])
            axes[1].set_title(f'Q_r CVaR (alpha={ALPHA})')"""

            dead_rgba = np.zeros((N_BINS, N_BINS, 4), dtype=np.uint8)
            dead_rgba[pred_action_grid] = [231, 76, 60, 200]
            dead_rgba[~pred_action_grid] = [255, 255, 255, 220]
            axes[2].imshow(np.transpose(dead_rgba, (1, 0, 2)), origin='lower', extent=[0, 10, 0, 10], interpolation='nearest', aspect='equal')
            axes[2].set_title('Predicted Dead-End Region')

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'cvar_heatmaps.svg'))
            plt.close()
            
            # 3. CONFUSION MATRIX & CLASSIFICATION METRICS
            rng = np.random.default_rng(seed)
            sampled_actions = rng.uniform(0.0, SIZE, size=(N_CONT_SAMPLES, 2)).astype(np.float32)
            
            ix, iy = nearest_discrete_indices(sampled_actions, action_bins)
            pred_sample = pred_action_grid[ix, iy]
            gt_sample = np.array([env._check_collision(a, env.danger_zones)[0] in ('death', 'dead_end') for a in sampled_actions], dtype=bool)
            
            TP = int((gt_sample & pred_sample).sum())
            TN = int((~gt_sample & ~pred_sample).sum())
            FP = int((~gt_sample & pred_sample).sum())
            FN = int((gt_sample & ~pred_sample).sum())
            
            cm = np.array([[TN, FP], [FN, TP]])
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Not Dead-End', 'Dead-End'])
            ax.set_yticklabels(['Not Dead-End', 'Dead-End'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'confusion_matrix.svg'))
            plt.close()
            
            #Save intermediary plot data
            np.savez_compressed(
                os.path.join(run_dir, 'intermediary_plot_data.npz'),
                ground_truth_img=img,
                cvar_d_grid=cvar_d_grid,
                cvar_r_grid=cvar_r_grid,
                pred_action_grid=pred_action_grid,
                confusion_matrix=cm,
                size=np.array(SIZE),
                alpha=np.array(ALPHA)
            )

            # 4. STORE METRICS
            precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
            recall = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
            
            xs_biou = np.linspace(0, SIZE, BIOU_RES, dtype=np.float32)
            ys_biou = np.linspace(0, SIZE, BIOU_RES, dtype=np.float32)
            gt_hires = np.array([[env._check_collision(np.array([x, y]), env.danger_zones)[0] in ('death', 'dead_end') for y in ys_biou] for x in xs_biou], dtype=bool)
            ix_h = np.abs(xs_biou[:, None] - action_bins[None, :]).argmin(axis=1)
            iy_h = np.abs(ys_biou[:, None] - action_bins[None, :]).argmin(axis=1)
            pred_hires = pred_action_grid[np.ix_(ix_h, iy_h)]
            
            biou = boundary_iou(gt_hires, pred_hires)
            
            metrics['n_dead_ends'].append(n)
            metrics['seed'].append(seed)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['biou'].append(biou)
            
    # ==========================================
    # SAVE DATA FOR PLOTTING
    # ==========================================
    output_file = "metrics_disc.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, output_file, indent=4)

    """plt.figure(figsize=(10, 6))
    for metric, label in zip(['precision', 'recall', 'f1', 'biou'], ['Precision', 'Recall', 'F1-Score', 'Boundary IoU']):
        means = []
        stds = []
        n_vals = sorted(list(set(metrics['n_dead_ends'])))
        for n in n_vals:
            vals = [metrics[metric][i] for i in range(len(metrics['n_dead_ends'])) if metrics['n_dead_ends'][i] == n]
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals))
        plt.errorbar(n_vals, means, yerr=stds, marker='o', label=label, capsize=4)
    
    plt.xlabel('Number of Dead-ends', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.title('Classification Metrics vs Number of Dead-ends', fontsize=14)
    plt.xticks(n_vals)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig('metrics_vs_dead_ends.svg', dpi=150)
    plt.close()
    """
    print("\nAll experiments finished. Consolidated metrics graph saved to 'metrics_vs_dead_ends.svg'.")

if __name__ == '__main__':
    run_experiment()