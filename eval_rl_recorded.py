"""
eval_rl_recorded.py

Evaluate a trained Continuous IQN at the **recorded clinical action** (same approach as
continuous_mimic_comparison.ipynb).  Produces ROC curves and AUC comparison against the
paper's discrete-action baselines.

Usage
-----
    python eval_rl_recorded.py -c iqn_continuous_mimic
    python eval_rl_recorded.py -c iqn_continuous_mimic --data test
    python eval_rl_recorded.py -c iqn_continuous_mimic --num_q_samples 64

This is intentionally separate from eval_rl.py (which uses grid-based dead-end volume
fractions).  This script evaluates Q only at the action actually taken, which is both
faster and produces a direct analogue of the discrete-IQN evaluation in initial_rl_results.ipynb.
"""

import os
import sys
import pickle

import click
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import auc as sklearn_auc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from rl_utils import ContinuousIQN_OfflineAgent
from analysis_utils import pre_flag_splitting, create_analysis_df, compute_auc

np.set_printoptions(suppress=True, linewidth=200, precision=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Paper baselines (Killian et al. 2023, discrete 25-action IQN, MIMIC-IV)
# ---------------------------------------------------------------------------
PAPER_AUC = {
    "DeD (DDQN, No CQL)":        0.6629,
    "DeD+CQL (DDQN, CQL)":       0.7687,
    "DistDeD-CQL (IQN, No CQL)": 0.7744,
    "DistDeD (IQN, CQL)":        0.7912,
}
PAPER_VAR_ALPHAS = np.linspace(0.05, 1.0, 20)
PAPER_DISTDED_MISSED_NS = np.array([
     4.790,  8.982, 10.180, 12.574, 13.772, 15.569, 17.964, 19.760, 21.557, 22.754,
    23.952, 26.347, 28.144, 30.539, 31.737, 34.131, 35.329, 36.527, 38.323, 41.916,
])
PAPER_DISTDED_MISSED_S = np.array([
    20.074, 30.439, 37.721, 44.071, 49.393, 54.435, 57.330, 60.598, 63.772, 66.013,
    68.627, 70.588, 72.549, 74.510, 76.657, 78.711, 79.925, 80.486, 81.979, 83.660,
])
PAPER_DED_MISSED_NS = 59.281
PAPER_DED_MISSED_S  = 88.982


# ---------------------------------------------------------------------------
# Model loading — auto-detects architecture from checkpoint weights
# ---------------------------------------------------------------------------

def _infer_arch_from_checkpoint(ckpt_state_dict, action_dim):
    """Read layer_size and state_dim from the saved weight shapes."""
    head_w = ckpt_state_dict["head.weight"]
    layer_size = int(head_w.shape[0])
    state_dim  = int(head_w.shape[1]) - action_dim
    return state_dim, layer_size


def load_model(ckpt_dir, sided_Q, params_base, device):
    """Load a ContinuousIQN_OfflineAgent, auto-detecting layer_size from checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, f"best_q_parameters{sided_Q}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt["rl_network_state_dict"]

    state_dim, layer_size = _infer_arch_from_checkpoint(sd, params_base["action_dim"])

    params = dict(params_base)
    params["input_dim"]          = state_dim
    params["num_q_hidden_units"] = layer_size

    agent = ContinuousIQN_OfflineAgent(
        state_size=state_dim,
        params=params,
        sided_Q=sided_Q,
        device=device,
    )
    agent.network.load_state_dict(sd)
    agent.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("validation_loss", [None])[-1]
    print(
        f"  Loaded {sided_Q:8s} net — epoch {epoch + 1 if isinstance(epoch, int) else epoch}, "
        f"layer_size={layer_size}, state_dim={state_dim}, "
        f"best val loss={val_loss:.6f}" if val_loss is not None else ""
    )
    return agent, state_dim


# ---------------------------------------------------------------------------
# State normalisation stats
# ---------------------------------------------------------------------------

def get_or_compute_norm_stats(ckpt_dir, data_dir, state_dim):
    """Load cached state norm stats or recompute from encoded_train.npz."""
    stats_path = os.path.join(ckpt_dir, "state_norm_stats.npz")
    if os.path.exists(stats_path):
        nc = np.load(stats_path)
        if nc["state_mean"].shape[0] == state_dim:
            print(f"  Loaded state_norm_stats.npz ({state_dim} dims)")
            return nc["state_mean"], nc["state_std"]
        print(f"  state_norm_stats.npz has wrong dim ({nc['state_mean'].shape[0]} ≠ {state_dim}) — recomputing")

    print("  Computing state norm stats from encoded_train.npz ...")
    tr = np.load(os.path.join(data_dir, "encoded_train.npz"), allow_pickle=True)
    total_count   = 0
    sum_s         = np.zeros(state_dim, dtype=np.float64)
    sum_sq_s      = np.zeros(state_dim, dtype=np.float64)
    for i in range(len(tr["states"])):
        T = int(tr["lengths"][i][0])
        s = tr["states"][i][:T].astype(np.float64)
        total_count  += T
        sum_s        += s.sum(axis=0)
        sum_sq_s     += (s ** 2).sum(axis=0)
    mean  = sum_s / total_count
    std   = np.sqrt(np.maximum(sum_sq_s / total_count - mean ** 2, 0)).clip(min=1e-8)
    np.savez(stats_path, state_mean=mean.astype(np.float32), state_std=std.astype(np.float32))
    print(f"  Saved to {stats_path}")
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Q-value evaluation at the recorded action (core of the comparison notebook)
# ---------------------------------------------------------------------------

def get_continuous_dn_rn_info(agent_dn, agent_rn, encoded_data, device, num_q_samples=64):
    """
    Evaluate D- and R-networks at the **recorded clinical action** for every valid step.

    Returns a dict suitable for analysis_utils.pre_flag_splitting (distributional=True).
    Shape of q_dn / q_rn per step: (num_q_samples, 1)  — single action, distributional.
    """
    agent_dn.eval()
    agent_rn.eval()

    n_total = len(encoded_data["states"])

    # Pass 1: count valid steps and gather metadata
    valid_trajs = []
    total_steps = 0
    for traj in range(n_total):
        traj_len   = int(encoded_data["lengths"][traj][0])
        terminal_r = float(encoded_data["rewards"][traj][traj_len - 1].flat[-1])
        if terminal_r in (-1.0, 1.0):
            category = -1 if terminal_r < 0 else 1
            valid_trajs.append((traj, traj_len, category))
            total_steps += traj_len

    n_kept = len(valid_trajs)
    print(f"  {n_kept}/{n_total} valid patients, {total_steps} total steps")
    if total_steps == 0:
        return {}

    first_traj = valid_trajs[0][0]
    s_dim = encoded_data["states"][first_traj].shape[1]
    a_sample = encoded_data["actions"][first_traj]
    a_shape  = (total_steps,) if a_sample.ndim == 1 else (total_steps, a_sample.shape[1])

    buf = {
        "traj":     np.empty(total_steps, dtype=np.int32),
        "step":     np.empty(total_steps, dtype=np.int32),
        "s":        np.empty((total_steps, s_dim), dtype=np.float32),
        "a":        np.empty(a_shape,              dtype=np.float32),
        "q_dn":     np.empty((total_steps, num_q_samples, 1), dtype=np.float32),
        "q_rn":     np.empty((total_steps, num_q_samples, 1), dtype=np.float32),
        "category": np.empty(total_steps, dtype=np.int8),
        "stay_id":  np.empty(total_steps, dtype=np.float64),
    }

    idx = 0
    with torch.no_grad():
        for traj, traj_len, category in valid_trajs:
            sid = encoded_data["stay_ids"][traj]
            states  = torch.from_numpy(encoded_data["states"][traj][:traj_len]).float().to(device)
            actions = torch.from_numpy(encoded_data["actions"][traj][:traj_len]).float().to(device)

            q_dn, _ = agent_dn.network(states, actions, num_q_samples)   # (T, N, 1)
            q_rn, _ = agent_rn.network(states, actions, num_q_samples)

            q_dn_np = np.sort(np.clip(q_dn.cpu().numpy(), -1.0, 0.0), axis=1).astype(np.float32)
            q_rn_np = np.sort(np.clip(q_rn.cpu().numpy(),  0.0, 1.0), axis=1).astype(np.float32)

            end = idx + traj_len
            buf["traj"][idx:end]     = traj
            buf["step"][idx:end]     = np.arange(traj_len)
            buf["s"][idx:end]        = states.cpu().numpy()
            buf["a"][idx:end]        = actions.cpu().numpy()
            buf["q_dn"][idx:end]     = q_dn_np
            buf["q_rn"][idx:end]     = q_rn_np
            buf["category"][idx:end] = category
            buf["stay_id"][idx:end]  = sid
            idx = end

    # Convert to list-of-arrays so pandas can store 2-D objects per cell
    buf["s"]    = list(buf["s"])
    buf["q_dn"] = list(buf["q_dn"])
    buf["q_rn"] = list(buf["q_rn"])
    if buf["a"].ndim > 1:
        buf["a"] = list(buf["a"])

    return buf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc(fpr, tpr, auc_vals, var_thresholds, save_path):
    palette = sns.color_palette("Greens", n_colors=20)
    fig, ax = plt.subplots(figsize=(7, 6))

    for ii in range(len(var_thresholds)):
        label = (f"Ours α={var_thresholds[ii]:.2f} (AUC={auc_vals[ii]:.3f})"
                 if ii % 4 == 0 else None)
        ax.plot(fpr[:, ii], tpr[:, ii], color=palette[ii], lw=1.2, alpha=0.85, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")

    for (name, val), xp, yp in zip(
        [("DeD (paper)", 0.6629), ("DistDeD (paper)", 0.7912)],
        [0.55, 0.35],
        [0.20, 0.55],
    ):
        ax.annotate(
            f"{name}\nAUC = {val:.4f}",
            xy=(xp, yp), fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    ax.set_xlabel("False Positive Rate (survivors flagged)")
    ax.set_ylabel("True Positive Rate (non-survivors flagged)")
    ax.set_title(
        f"ROC — Continuous IQN (recorded action)\n"
        f"Mean AUC = {auc_vals.mean():.4f}  |  Best AUC = {auc_vals.max():.4f}"
        f"  (α={var_thresholds[auc_vals.argmax()]:.2f})"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_auc_vs_alpha(auc_vals, var_thresholds, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(var_thresholds, auc_vals, "o-", color="#1a9641", lw=2, markersize=5,
            label=f"Ours — Cont. IQN+CQL (mean={auc_vals.mean():.4f})")

    styles = [
        ("DeD (DDQN, No CQL)",         0.6629, "black",   "--"),
        ("DeD+CQL (DDQN, CQL)",        0.7687, "#d7191c", ":"),
        ("DistDeD-CQL (IQN, No CQL)", 0.7744, "#fdae61", "-."),
        ("DistDeD (IQN, CQL)",         0.7912, "#2c7bb6", "--"),
    ]
    for label, val, col, ls in styles:
        ax.axhline(val, color=col, linestyle=ls, lw=1.5, label=f"{label} = {val:.4f} [paper]")

    ax.set_xlabel("VaR Threshold (α)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("AUC vs CVaR α — Continuous IQN vs Paper Baselines", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1.05); ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_training_curves(ckpt_dir, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sided_Q, title, col in zip(
        axes,
        ["negative", "positive"],
        ["D-Network (negative rewards)", "R-Network (positive rewards)"],
        ["#2166ac", "#d6604d"],
    ):
        loss_path = os.path.join(ckpt_dir, f"q_losses_{sided_Q}.npy")
        ckpt_path = os.path.join(ckpt_dir, f"q_parameters{sided_Q}.pt")
        if os.path.exists(loss_path):
            train_loss = np.load(loss_path)
            ax.plot(np.arange(1, len(train_loss) + 1), train_loss,
                    color=col, lw=2, label="Train loss")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            val_loss = ckpt.get("validation_loss", [])
            if val_loss:
                ax.plot(
                    np.linspace(1, len(np.load(loss_path)) if os.path.exists(loss_path) else len(val_loss),
                                len(val_loss)),
                    val_loss, color=col, linestyle="--", lw=1.5, label="Val loss",
                )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Continuous IQN Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_missed_trajs(our_missed_ns, our_missed_s, var_thresholds, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, our_vals, paper_vals, paper_ded, title, better in zip(
        axes,
        [our_missed_ns, our_missed_s],
        [PAPER_DISTDED_MISSED_NS, PAPER_DISTDED_MISSED_S],
        [PAPER_DED_MISSED_NS, PAPER_DED_MISSED_S],
        ["Non-survivors Missed (lower = better)", "Survivors Missed (higher = fewer false alarms)"],
        ["lower", "higher"],
    ):
        ax.plot(var_thresholds, our_vals, "o-", color="#1a9641", lw=2, markersize=5,
                label="Ours — Cont. IQN+CQL")
        ax.plot(PAPER_VAR_ALPHAS, paper_vals, "s--", color="#2c7bb6", lw=1.5,
                markersize=4, label="DistDeD (paper)")
        ax.axhline(paper_ded, color="black", linestyle="--", lw=1.5,
                   label=f"DeD (paper) = {paper_ded:.1f}%")
        ax.set_xlabel("VaR Threshold (α)", fontsize=11)
        ax.set_ylabel("% Trajectories Missed", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1.05)
    fig.suptitle("% Trajectories with No Flag Raised  (δD = −0.5, δR = 0.5)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "-c", default="iqn_continuous_mimic", help="Config name (without .yaml)")
@click.option("--data",   "-d", default="test",                 help="Data split: train/validation/test")
@click.option("--num_q_samples", "-n", default=64, type=int,   help="Quantile samples for distributional eval")
@click.option("--smoke_test", is_flag=True, default=False,      help="Run on first 10 trajectories only")
def run(config, data, num_q_samples, smoke_test):
    """Evaluate Continuous IQN at the recorded clinical action and produce ROC/AUC plots."""

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params   = yaml.safe_load(open(os.path.join(dir_path, f"configs/{config}.yaml")))

    print("=" * 60)
    print(f"Config      : {config}")
    print(f"Data split  : {data}")
    print(f"Num Q samples: {num_q_samples}")
    print(f"Smoke test  : {smoke_test}")
    print(f"Device      : {device}")
    print("=" * 60)

    ckpt_dir = params["checkpoint_fname"]
    data_dir = params["data_dir"]
    action_dim = int(params["action_dim"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 1. Load models (auto-detect architecture from checkpoint) ─────────
    print("\n[1/6] Loading models ...")
    agent_dn, state_dim_dn = load_model(ckpt_dir, "negative", params, device)
    agent_rn, state_dim_rn = load_model(ckpt_dir, "positive", params, device)
    if state_dim_dn != state_dim_rn:
        print(
            f"  Warning: D-net state_dim={state_dim_dn} ≠ R-net state_dim={state_dim_rn}. "
            "Using D-net dim for data loading."
        )
    state_dim = state_dim_dn

    # ── 2. State norm stats ───────────────────────────────────────────────
    print("\n[2/6] State normalisation stats ...")
    state_mean, state_std = get_or_compute_norm_stats(ckpt_dir, data_dir, state_dim)

    # Inject norm stats into both agents (they use these internally for _normalise())
    for agent in (agent_dn, agent_rn):
        dev = torch.device(agent.device) if isinstance(agent.device, str) else agent.device
        agent.network._s_mean = torch.FloatTensor(state_mean).to(dev)
        agent.network._s_scale = torch.FloatTensor(state_std).to(dev)

    # ── 3. Load encoded data ──────────────────────────────────────────────
    print(f"\n[3/6] Loading encoded_{data}.npz ...")
    encoded = np.load(os.path.join(data_dir, f"encoded_{data}.npz"), allow_pickle=True)
    enc_state_dim = int(encoded["states"][0].shape[1])
    if enc_state_dim != state_dim:
        print(
            f"  Warning: encoded state_dim={enc_state_dim} ≠ checkpoint state_dim={state_dim}. "
            "The states will be passed as-is — check your encoded data."
        )
    print(f"  {len(encoded['states'])} trajectories, state_dim={enc_state_dim}")

    if smoke_test:
        print("  [SMOKE TEST] limiting to first 10 trajectories")
        class _Subset:
            def __init__(self, d, n):
                self._d = d
                self._n = n
                self.files = d.files
            def __getitem__(self, k):
                return self._d[k][:self._n]
        encoded = _Subset(encoded, 10)

    # ── 4. Evaluate networks at recorded actions ──────────────────────────
    print(f"\n[4/6] Evaluating Q at recorded actions (num_q_samples={num_q_samples}) ...")
    cache_path = os.path.join(ckpt_dir, f"value_data_recorded_{data}.pkl")

    if os.path.exists(cache_path) and not smoke_test:
        print(f"  Loading cached value_data from {cache_path}")
        with open(cache_path, "rb") as f:
            value_data = pickle.load(f)
    else:
        value_data = get_continuous_dn_rn_info(agent_dn, agent_rn, encoded, device, num_q_samples)
        if not smoke_test:
            with open(cache_path, "wb") as f:
                pickle.dump(value_data, f)
            print(f"  Cached to {cache_path}")

    cats = pd.DataFrame({"traj": value_data["traj"], "cat": value_data["category"]}).drop_duplicates("traj")["cat"]
    n_survivors    = int((cats ==  1).sum())
    n_nonsurvivors = int((cats == -1).sum())
    print(f"  Survivors: {n_survivors},  Non-survivors: {n_nonsurvivors}")

    # ── 5. Pre-flag splitting → CVaR per alpha ────────────────────────────
    print("\n[5/6] Computing CVaR and flags ...")
    VaR_thresholds = np.round(np.linspace(0.05, 1.0, num=20), decimals=2)
    results = pre_flag_splitting(value_data, VaR_thresholds, distributional=True)

    n_surv_res    = len(results["survivors"]["dn_q_selected_action_traj"])
    n_nonsurv_res = len(results["nonsurvivors"]["dn_q_selected_action_traj"])

    surv_df, nonsurv_df = create_analysis_df(results, n_surv_res, n_nonsurv_res)

    fpr, tpr, auc_out = compute_auc(surv_df, nonsurv_df, n_surv_res, n_nonsurv_res, iqn_size=20)
    auc_vals = np.asarray(auc_out[0])

    print(f"\n  AUC per CVaR alpha:")
    print(f"  {'Alpha':>6}  {'AUC':>8}")
    print("  " + "-" * 18)
    for alpha, aval in zip(VaR_thresholds, auc_vals):
        print(f"  {alpha:>6.2f}  {aval:>8.4f}")
    print(f"\n  Best AUC : {auc_vals.max():.4f}  (alpha = {VaR_thresholds[auc_vals.argmax()]:.2f})")
    print(f"  Mean AUC : {auc_vals.mean():.4f}")

    # % missed at fixed operating point (δD=-0.5, δR=0.5 → threshold index 500)
    FIXED_THR_IDX = 500
    from analysis_utils import compare_flag_range

    missed_ns = np.zeros(20)
    for traj in range(n_nonsurv_res):
        dt = nonsurv_df[nonsurv_df.traj == traj]
        flags = np.stack(dt.apply(compare_flag_range, axis=1).values)
        for ii in range(20):
            if not np.any(flags[:, FIXED_THR_IDX, ii]):
                missed_ns[ii] += 1
    missed_ns = 100.0 * missed_ns / max(n_nonsurv_res, 1)

    missed_s = np.zeros(20)
    for traj in range(n_surv_res):
        dt = surv_df[surv_df.traj == traj]
        flags = np.stack(dt.apply(compare_flag_range, axis=1).values)
        for ii in range(20):
            if not np.any(flags[:, FIXED_THR_IDX, ii]):
                missed_s[ii] += 1
    missed_s = 100.0 * missed_s / max(n_surv_res, 1)

    print(f"\n  % non-survivors missed: {missed_ns.min():.1f}% – {missed_ns.max():.1f}%")
    print(f"  % survivors   missed  : {missed_s.min():.1f}% – {missed_s.max():.1f}%")

    # ── 6. Save plots and results ─────────────────────────────────────────
    print("\n[6/6] Saving plots ...")

    suffix = f"_{data}" if data != "test" else ""

    plot_training_curves(ckpt_dir, os.path.join(ckpt_dir, "rec_fig1_training_curves.png"))
    plot_roc(fpr, tpr, auc_vals, VaR_thresholds,
             os.path.join(ckpt_dir, f"rec_fig2_roc_curves{suffix}.png"))
    plot_auc_vs_alpha(auc_vals, VaR_thresholds,
                      os.path.join(ckpt_dir, f"rec_fig3_auc_vs_alpha{suffix}.png"))
    plot_missed_trajs(missed_ns, missed_s, VaR_thresholds,
                      os.path.join(ckpt_dir, f"rec_fig4_missed_trajs{suffix}.png"))

    # Save AUC table
    auc_df = pd.DataFrame({"alpha": VaR_thresholds, "auc": auc_vals})
    auc_csv = os.path.join(ckpt_dir, f"rec_auc_table{suffix}.csv")
    auc_df.to_csv(auc_csv, index=False)
    print(f"  Saved: {auc_csv}")

    # Summary
    print("\n" + "=" * 60)
    print(" Continuous IQN — Recorded-Action Evaluation Summary")
    print("=" * 60)
    print(f" Config            : {config}")
    print(f" Data split        : {data}")
    print(f" Survivors         : {n_survivors}")
    print(f" Non-survivors     : {n_nonsurvivors}")
    print(f" Best AUC          : {auc_vals.max():.4f}  (α={VaR_thresholds[auc_vals.argmax()]:.2f})")
    print(f" Mean AUC          : {auc_vals.mean():.4f}")
    print(f" Paper DistDeD     : 0.7912")
    print(f" Paper DeD         : 0.6629")
    print("=" * 60)
    print(f"\nOutputs in: {ckpt_dir}")


if __name__ == "__main__":
    run()
