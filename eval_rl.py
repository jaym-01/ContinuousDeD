## IMPORTS
import os, sys, time
import random
import pickle
import click
import yaml

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch

from rl_utils import DQN_Agent, IQN_Agent, ContinuousIQN_OfflineAgent
from plot_utils import plot_value_hists
from analysis_utils import get_dn_rn_info, create_analysis_df
from boundary_tracing import dead_end_volume_fraction_multi_alpha, dead_end_volume_fraction_grid_batch, grid_cvar_batch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
##     HELPER FUNCTIONS
#############################

def load_best_rl(params, sided_Q, device):
    best_rl_path = os.path.join(params['checkpoint_fname'], f"best_q_parameters{sided_Q}.pt")
    best_rl_checkpoint = torch.load(best_rl_path, map_location=device, weights_only=False)

    # Align model input size with checkpoint to avoid silent shape mismatches.
    if 'rl_network_state_dict' in best_rl_checkpoint and 'head.weight' in best_rl_checkpoint['rl_network_state_dict']:
        ckpt_in_dim = best_rl_checkpoint['rl_network_state_dict']['head.weight'].shape[1]
        # For ContinuousIQN, head takes [state ∥ action], so head_dim = state_dim + action_dim.
        # Subtract action_dim to recover the pure state_dim.
        if params.get('model') == 'ContinuousIQN':
            ckpt_in_dim -= params.get('action_dim', 0)
        params['input_dim'] = int(ckpt_in_dim)

    # Initialize the model
    if params['model'] == 'DQN':
        model = DQN_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    elif params['model'] == 'IQN':
        model = IQN_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    elif params['model'] == 'ContinuousIQN':
        model = ContinuousIQN_OfflineAgent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    else:
        raise NotImplementedError('The provided model type has not yet been defined, please use DQN, IQN, or ContinuousIQN')

    # Load the best performing parameters (based on validation loss) into the model
    model.network.load_state_dict(best_rl_checkpoint['rl_network_state_dict'])
    model.eval()
    print(f"{sided_Q.capitalize()} Q-Network loaded")
    return model


def _get_checkpoint_input_dim(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('rl_network_state_dict', {})
    if 'head.weight' not in state_dict:
        return None
    return int(state_dict['head.weight'].shape[1])


def _adjust_state_dim(states, target_dim, label="states"):
    """Pad or truncate the last dimension to match target_dim."""
    cur_dim = states.shape[-1]
    if cur_dim == target_dim:
        return states
    if cur_dim < target_dim:
        pad = target_dim - cur_dim
        print(
            f"Warning: {label} has {cur_dim} features; padding with {pad} zeros to match {target_dim}.",
            flush=True,
        )
        return np.pad(states, ((0, 0), (0, 0), (0, pad)), mode="constant")
    # cur_dim > target_dim
    drop = cur_dim - target_dim
    print(
        f"Warning: {label} has {cur_dim} features; truncating last {drop} to match {target_dim}.",
        flush=True,
    )
    return states[..., :target_dim]


def _coerce_encoded_state_dim(encoded_data, target_dim, label):
    """Return a dict-like encoded_data with states adjusted to target_dim."""
    states = encoded_data['states']
    if states.shape[-1] == target_dim:
        return encoded_data

    # Materialize arrays so we can safely replace states.
    data_dict = {k: encoded_data[k] for k in encoded_data.files}
    data_dict['states'] = _adjust_state_dim(data_dict['states'], target_dim, label=label)
    return data_dict


def _resolve_action_bounds(params):
    action_dim = params.get('action_dim', 2)
    action_low = params.get('action_low', [0.0] * action_dim)
    action_high = params.get('action_high', [1.0] * action_dim)
    if len(action_low) != action_dim or len(action_high) != action_dim:
        raise ValueError("action_low/action_high length must match action_dim")
    return action_low, action_high


def get_continuous_dead_end_data(
    qnet_dn,
    encoded_data,
    device,
    params,
    bnd_M=20,
    bnd_alphas=(0.1,),
    bnd_delta_D=-0.5,
    bnd_h0=0.05,
    bnd_eps_tol=1e-4,
    bnd_eps_close=0.02,
    bnd_eps_dup=0.02,
    bnd_eta=1.5,
    bnd_C_max=10,
    bnd_num_tau=64,
    checkpoint_path=None,
    checkpoint_every=100,
    log_every_traj=10,
    log_every_state=0,
):
    """Compute dead-end volume fraction f_D per state for all alpha values.

    Uses predictor-corrector boundary tracing (Algorithm 1) for continuous IQN.
    Supports checkpoint/resume: saves progress every `checkpoint_every` valid
    trajectories to `checkpoint_path` (per-trajectory granularity).

    Returns
    -------
    data_by_alpha : dict {alpha: pd.DataFrame}
    """
    n_total = len(encoded_data['states'])
    action_low, action_high = _resolve_action_bounds(params)
    alphas_f = [float(a) for a in bnd_alphas]

    n_valid = sum(
        1
        for i in range(n_total)
        if float(encoded_data['rewards'][i][int(encoded_data['lengths'][i].flat[0]) - 1].flat[-1])
        in (-1.0, 1.0)
    )

    rows_by_alpha = {a: [] for a in alphas_f}
    resume_from_traj = 0
    n_done = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        rows_by_alpha = ckpt['rows_by_alpha']
        resume_from_traj = ckpt['next_traj']
        n_done = ckpt['n_done']
        print(
            f"  Resuming at trajectory index {resume_from_traj} ({n_done}/{n_valid} done)",
            flush=True,
        )

    print(
        "Computing dead-end volume fractions via predictor-corrector tracing "
        f"(M={bnd_M}, {len(alphas_f)} alpha levels) — {n_valid} valid trajectories...",
        flush=True,
    )

    for traj in range(resume_from_traj, n_total):
        traj_len = int(encoded_data['lengths'][traj].flat[0])
        traj_r = encoded_data['rewards'][traj][:traj_len]
        terminal_r = float(traj_r[-1].flat[-1])
        if terminal_r not in (-1.0, 1.0):
            continue

        print(f"  starting traj {traj} (len={traj_len})", flush=True)

        category = -1 if terminal_r < 0 else 1
        traj_sid = encoded_data['stay_ids'][traj]
        states = encoded_data['states'][traj][:traj_len]

        for t in range(traj_len):
            f_D_per_alpha = dead_end_volume_fraction_multi_alpha(
                states[t],
                qnet_dn,
                alphas=alphas_f,
                delta_D=bnd_delta_D,
                M=bnd_M,
                h0=bnd_h0,
                eps_tol=bnd_eps_tol,
                eps_close=bnd_eps_close,
                eps_dup=bnd_eps_dup,
                eta=bnd_eta,
                C_max=bnd_C_max,
                action_low=action_low,
                action_high=action_high,
                num_tau=bnd_num_tau,
            )
            if log_every_state and (t % log_every_state == 0):
                print(
                    f"  traj {traj} step {t}/{traj_len} (alpha={alphas_f[0]:.2f}..{alphas_f[-1]:.2f})",
                    flush=True,
                )

            for alpha in alphas_f:
                f_d = float(f_D_per_alpha[alpha])
                rows_by_alpha[alpha].append(
                    {
                        'traj': traj,
                        'step': t,
                        'f_D': f_d,
                        'v_dn': -f_d,
                        'v_rn': 1.0 - f_d,
                        'category': category,
                        'stay_id': traj_sid,
                    }
                )

        n_done += 1
        print(f"  finished traj {traj} ({n_done}/{n_valid})", flush=True)
        if log_every_traj and (n_done % log_every_traj == 0):
            print(f"  {n_done}/{n_valid} trajectories processed", flush=True)

        if checkpoint_path and n_done % checkpoint_every == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(
                    {
                        'rows_by_alpha': rows_by_alpha,
                        'next_traj': traj + 1,
                        'n_done': n_done,
                    },
                    f,
                )
            print(f"  [checkpoint saved at {n_done}/{n_valid}]", flush=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed (run complete).")

    return {a: pd.DataFrame(rows) for a, rows in rows_by_alpha.items()}

def get_continuous_dead_end_data_grid(
    qnet_dn,
    qnet_rn,
    encoded_data,
    device,
    params,
    alphas,
    delta_D=-0.7,
    delta_R=0.5,
    M=10,
    log_every_traj=10,
):
    """Grid-based dead-end detection for continuous action-space IQN.

    For each state computes:
      f_D = fraction of M×M grid actions where CVaR_α(Q_D) < delta_D
      f_R = fraction of M×M grid actions where CVaR_α(Q_R) < delta_R

    Then stores v_dn = -f_D  and  v_rn = 1 - f_R, giving independent
    D and R signals for the downstream flag condition.

    Returns
    -------
    data_by_alpha : dict {alpha: pd.DataFrame}
        Columns: traj, step, f_D, f_R, v_dn, v_rn, category, stay_id
    """
    n_total    = len(encoded_data['states'])
    action_low  = params.get('action_low',  [0.0] * params.get('action_dim', 2))
    action_high = params.get('action_high', [1.0] * params.get('action_dim', 2))
    alphas_f   = [float(a) for a in alphas]
    num_tau    = params.get('bnd_num_tau', params.get('num_iqn_samples_est', 64))

    n_valid = sum(
        1 for i in range(n_total)
        if float(encoded_data['rewards'][i][int(encoded_data['lengths'][i].flat[0]) - 1].flat[-1])
        in (-1.0, 1.0)
    )

    rows_by_alpha = {a: [] for a in alphas_f}
    n_done = 0

    print(
        f"Grid dead-end detection (M={M}×{M}, {len(alphas_f)} alpha levels, "
        f"delta_D={delta_D}, delta_R={delta_R}) — {n_valid} valid trajectories...",
        flush=True,
    )

    for traj in range(n_total):
        traj_len   = int(encoded_data['lengths'][traj].flat[0])
        traj_r     = encoded_data['rewards'][traj][:traj_len]
        terminal_r = float(traj_r[-1].flat[-1])
        if terminal_r not in (-1.0, 1.0):
            continue

        category = -1 if terminal_r < 0 else 1
        traj_sid = encoded_data['stay_ids'][traj]
        states   = encoded_data['states'][traj][:traj_len]  # (T, state_dim)

        # Single batched forward pass per network for all T states × M² actions
        fD = dead_end_volume_fraction_grid_batch(
            states, qnet_dn, alphas_f,
            delta_D=delta_D, M=M,
            action_low=action_low, action_high=action_high,
            num_tau=num_tau,
        )  # (T, n_alphas)

        fR = dead_end_volume_fraction_grid_batch(
            states, qnet_rn, alphas_f,
            delta_D=delta_R, M=M,
            action_low=action_low, action_high=action_high,
            num_tau=num_tau,
        )  # (T, n_alphas)

        for j, alpha in enumerate(alphas_f):
            for t in range(traj_len):
                f_d = float(fD[t, j])
                f_r = float(fR[t, j])
                rows_by_alpha[alpha].append({
                    'traj':     traj,
                    'step':     t,
                    'f_D':      f_d,
                    'f_R':      f_r,
                    'v_dn':     -f_d,
                    'v_rn':     1.0 - f_r,
                    'category': category,
                    'stay_id':  traj_sid,
                })

        n_done += 1
        if log_every_traj and (n_done % log_every_traj == 0):
            print(f"  {n_done}/{n_valid} trajectories processed", flush=True)

    return {a: pd.DataFrame(rows) for a, rows in rows_by_alpha.items()}


def _probe_q_percentile(qnet, encoded_data, params, percentile=25, n_samples=500, alpha=0.1):
    """Sample n_samples random (state, action) pairs and return the p-th percentile of CVaR_alpha(Q).

    Used to auto-calibrate delta_D so it falls inside the model's learned Q range,
    which is necessary for the predictor-corrector to find a non-trivial boundary and
    for f_D to vary between dead-end and recoverable states.
    """
    device = qnet.device
    action_low  = params.get('action_low',  [0.0] * params.get('action_dim', 2))
    action_high = params.get('action_high', [1.0] * params.get('action_dim', 2))
    num_tau     = params.get('num_iqn_samples_est', 64)
    n_traj      = len(encoded_data['states'])
    rng         = np.random.RandomState(0)

    states_list  = []
    actions_list = []
    n_collected  = 0
    for _ in range(n_samples * 10):
        traj = rng.randint(n_traj)
        traj_len = int(encoded_data['lengths'][traj].flat[0])
        if traj_len == 0:
            continue
        t = rng.randint(traj_len)
        states_list.append(encoded_data['states'][traj][t])
        a = np.array([rng.uniform(lo, hi) for lo, hi in zip(action_low, action_high)], dtype=np.float32)
        actions_list.append(a)
        n_collected += 1
        if n_collected >= n_samples:
            break

    states_t  = torch.as_tensor(np.stack(states_list,  axis=0).astype(np.float32), device=device)
    actions_t = torch.as_tensor(np.stack(actions_list, axis=0).astype(np.float32), device=device)

    qnet.eval()
    with torch.no_grad():
        quantiles, _ = qnet.network(states_t, actions_t, num_tau)  # (N, num_tau, 1)
    sorted_q = quantiles.squeeze(-1).sort(dim=1)[0]  # (N, num_tau)
    k = max(1, round(alpha * num_tau))
    cvar = sorted_q[:, :k].mean(dim=1).cpu().numpy()  # (N,)

    val = float(np.percentile(cvar, percentile))
    print(
        f"Q-probe ({n_collected} pairs, α={alpha}): "
        f"min={cvar.min():.4f}  p25={np.percentile(cvar,25):.4f}  "
        f"median={np.median(cvar):.4f}  p75={np.percentile(cvar,75):.4f}  max={cvar.max():.4f}",
        flush=True,
    )
    print(f"Auto delta_D = p{percentile} = {val:.4f}", flush=True)
    return val


def _assemble_pc_results(data_by_alpha, alphas_f):
    """Convert predictor-corrector long-form DataFrames into the standard results dict.

    data_by_alpha : {alpha: pd.DataFrame} with columns traj, step, f_D, f_R, v_dn, v_rn, category, stay_id
    Returns results dict with dn_q_selected_action_traj etc., each list of (T, n_alphas) arrays.
    """
    first_df = data_by_alpha[alphas_f[0]]
    n_alphas = len(alphas_f)

    non_survivor_trajs = sorted(first_df[first_df.category == -1].traj.unique().tolist())
    survivor_trajs     = sorted(first_df[first_df.category == 1].traj.unique().tolist())

    results = {'nonsurvivors': {}, 'survivors': {}}

    for traj_type, trajectories in [('nonsurvivors', non_survivor_trajs), ('survivors', survivor_trajs)]:
        vdn_list = []
        vrn_list = []
        sid_list = []

        for traj in trajectories:
            traj_ref  = first_df[first_df.traj == traj].sort_values('step')
            T         = len(traj_ref)
            vdn_stack = np.zeros((T, n_alphas), dtype=np.float32)
            vrn_stack = np.zeros((T, n_alphas), dtype=np.float32)

            for j, alpha in enumerate(alphas_f):
                df_a    = data_by_alpha[alpha]
                traj_df = df_a[df_a.traj == traj].sort_values('step')
                vdn_stack[:, j] = traj_df['v_dn'].values.astype(np.float32)
                vrn_stack[:, j] = traj_df['v_rn'].values.astype(np.float32)

            vdn_list.append(vdn_stack)
            vrn_list.append(vrn_stack)
            sid_list.append(traj_ref['stay_id'].iloc[0])

        results[traj_type] = {
            'dn_q_selected_action_traj': vdn_list,
            'rn_q_selected_action_traj': vrn_list,
            'dn_v_median_traj':          vdn_list,
            'rn_v_median_traj':          vrn_list,
            'stay_ids':                  sid_list,
        }

    return results


def _probe_q_percentiles(qnet, encoded_data, params, percentiles, n_samples=500, alpha=0.1):
    """Like _probe_q_percentile but returns all percentile values in one pass.

    Avoids N_candidates separate 500-sample forward passes by doing one pass
    and reading off all requested percentiles from the same CVaR distribution.
    """
    device = qnet.device
    action_low  = params.get('action_low',  [0.0] * params.get('action_dim', 2))
    action_high = params.get('action_high', [1.0] * params.get('action_dim', 2))
    num_tau     = params.get('num_iqn_samples_est', 64)
    n_traj      = len(encoded_data['states'])
    rng         = np.random.RandomState(0)

    states_list, actions_list = [], []
    for _ in range(n_samples * 10):
        traj     = rng.randint(n_traj)
        traj_len = int(encoded_data['lengths'][traj].flat[0])
        if traj_len == 0:
            continue
        t = rng.randint(traj_len)
        states_list.append(encoded_data['states'][traj][t])
        actions_list.append(np.array(
            [rng.uniform(lo, hi) for lo, hi in zip(action_low, action_high)], dtype=np.float32
        ))
        if len(states_list) >= n_samples:
            break

    states_t  = torch.as_tensor(np.stack(states_list,  0).astype(np.float32), device=device)
    actions_t = torch.as_tensor(np.stack(actions_list, 0).astype(np.float32), device=device)

    qnet.eval()
    with torch.no_grad():
        quantiles, _ = qnet.network(states_t, actions_t, num_tau)
    sorted_q = quantiles.squeeze(-1).sort(dim=1)[0]
    k    = max(1, round(alpha * num_tau))
    cvar = sorted_q[:, :k].mean(dim=1).cpu().numpy()

    print(
        f"Q-probe ({len(states_list)} pairs, α={alpha}): "
        f"min={cvar.min():.4f}  p25={np.percentile(cvar,25):.4f}  "
        f"median={np.median(cvar):.4f}  p75={np.percentile(cvar,75):.4f}  max={cvar.max():.4f}",
        flush=True,
    )
    out = {pct: float(np.percentile(cvar, pct)) for pct in percentiles}
    for pct, val in out.items():
        print(f"  p{pct} = {val:.4f}", flush=True)
    return out


def _select_delta_D_grid(qnet_dn, qnet_rn, encoded_data, params, candidates, alphas_f, delta_D_rn, M=5):
    """Pre-screen delta_D candidates using a fast grid approximation.

    For each candidate delta_D, computes grid-based f_D / f_R in a single
    batched forward pass per trajectory (vs. the full predictor-corrector which
    does O(T × N_PC_iters) passes per trajectory). Typically 100–1000× cheaper.

    The R-net fR values are cached once (delta_D_rn is fixed across candidates)
    so only N_candidates D-net forward passes are needed across all trajectories.

    Parameters
    ----------
    candidates : dict {percentile_label: delta_D}  e.g. {10: -0.12, 25: -0.08, ...}

    Returns
    -------
    best_delta_D : float
    best_label   : str  e.g. 'p25'
    """
    action_low  = params.get('action_low',  [0.0] * params.get('action_dim', 2))
    action_high = params.get('action_high', [1.0] * params.get('action_dim', 2))
    num_tau     = params.get('num_iqn_samples_est', 64)
    n_total     = len(encoded_data['states'])

    # Identify valid trajectories and cache R-net fR (fixed across all D-net candidates)
    print(f"  [grid pre-screen] Caching R-net grid pass (delta_D_rn={delta_D_rn:.4f})...", flush=True)
    valid_trajs  = []
    vrn_cache    = {}
    label_cache  = {}
    for traj in range(n_total):
        traj_len   = int(encoded_data['lengths'][traj].flat[0])
        traj_r     = encoded_data['rewards'][traj][:traj_len]
        terminal_r = float(traj_r[-1].flat[-1])
        if terminal_r not in (-1.0, 1.0):
            continue
        states = encoded_data['states'][traj][:traj_len]
        fR = dead_end_volume_fraction_grid_batch(
            states, qnet_rn, alphas_f,
            delta_D=delta_D_rn, M=M,
            action_low=action_low, action_high=action_high,
            num_tau=num_tau,
        )
        vrn_cache[traj]   = 1.0 - fR
        label_cache[traj] = -1 if terminal_r < 0 else 1
        valid_trajs.append(traj)

    best_auc     = -1.0
    best_label   = None
    best_delta_D = None

    print(f"  [grid pre-screen] Testing {len(candidates)} candidates over {len(valid_trajs)} trajectories...", flush=True)
    for pct, delta_D_dn in candidates.items():
        label = f'p{pct}'
        ns_vdn, ns_vrn, s_vdn, s_vrn = [], [], [], []

        for traj in valid_trajs:
            traj_len = int(encoded_data['lengths'][traj].flat[0])
            states   = encoded_data['states'][traj][:traj_len]
            fD = dead_end_volume_fraction_grid_batch(
                states, qnet_dn, alphas_f,
                delta_D=delta_D_dn, M=M,
                action_low=action_low, action_high=action_high,
                num_tau=num_tau,
            )
            vdn = -fD
            vrn = vrn_cache[traj]
            if label_cache[traj] == -1:
                ns_vdn.append(vdn); ns_vrn.append(vrn)
            else:
                s_vdn.append(vdn);  s_vrn.append(vrn)

        mock = {
            'nonsurvivors': {'dn_q_selected_action_traj': ns_vdn, 'rn_q_selected_action_traj': ns_vrn},
            'survivors':    {'dn_q_selected_action_traj': s_vdn,  'rn_q_selected_action_traj': s_vrn},
        }
        auc = _quick_auc(mock)
        print(f"    {label}: delta_D={delta_D_dn:.4f}  AUC={auc:.4f}", flush=True)

        if auc > best_auc:
            best_auc     = auc
            best_label   = label
            best_delta_D = delta_D_dn

    print(f"  [grid pre-screen] Best: {best_label}={best_delta_D:.4f}  (AUC={best_auc:.4f})", flush=True)
    return best_delta_D, best_label


def _quick_auc(results, alpha_idx=0, n_thr=201):
    """Estimate AUC at one alpha level for delta_D sweep selection.

    Uses flag condition: v_dn < thr-1.0 AND v_rn < thr.
    Vectorised over all timesteps and thresholds simultaneously.
    """
    thresholds = np.linspace(0.0, 1.0, n_thr, dtype=np.float32)

    def _ever_flagged(group):
        flags = []
        for vdn_arr, vrn_arr in zip(
            results[group]['dn_q_selected_action_traj'],
            results[group]['rn_q_selected_action_traj'],
        ):
            vdn = np.asarray(vdn_arr, dtype=np.float32)
            vrn = np.asarray(vrn_arr, dtype=np.float32)
            if vdn.ndim > 1: vdn = vdn[:, alpha_idx]
            if vrn.ndim > 1: vrn = vrn[:, alpha_idx]
            f = (vdn[:, None] < (thresholds[None, :] - 1.0)) & (vrn[:, None] < thresholds[None, :])
            flags.append(f.any(axis=0))  # (n_thr,)
        return np.stack(flags, axis=0)   # (n_traj, n_thr)

    ns = _ever_flagged('nonsurvivors')
    s  = _ever_flagged('survivors')
    tpr = ns.mean(axis=0)
    fpr = s.mean(axis=0)
    idx = np.argsort(fpr)
    return float(np.trapz(tpr[idx], fpr[idx]))


def get_continuous_dead_end_data_grid_cvar(
    qnet_dn,
    qnet_rn,
    encoded_data,
    device,
    params,
    alphas,
    M=10,
    log_every_traj=10,
):
    """Grid-based dead-end detection storing max CVaR over grid actions.

    Replaces get_continuous_dead_end_data_grid for ROC computation.

    Instead of computing f_D = fraction of actions below a fixed delta_D,
    this stores the best-case CVaR directly:

        v_dn[s, α] = max_a CVaR_α(Q_D(s,a))  ∈ [-1, 0]
        v_rn[s, α] = max_a CVaR_α(Q_R(s,a))  ∈ [0,  1]

    Why this fixes AUC=0:
    - The old approach requires delta_D to fall inside the model's Q-value range.
      If delta_D=-0.7 but Q_D ∈ [-0.3, 0], f_D=0 everywhere → no signal.
    - Storing max CVaR directly gives a continuous signal whose range always
      matches the model's outputs. Sweeping the downstream flagging threshold
      is then equivalent to sweeping delta_D across the full Q-value range,
      so the ROC is well-defined regardless of where Q-values are concentrated.

    Returns
    -------
    results : dict with 'survivors' and 'nonsurvivors' sub-dicts, each containing:
        dn_q_selected_action_traj : list[ndarray(T, n_alphas)]
        rn_q_selected_action_traj : list[ndarray(T, n_alphas)]
        dn_v_median_traj          : list[ndarray(T, n_alphas)]  (same as above)
        rn_v_median_traj          : list[ndarray(T, n_alphas)]  (same as above)
        stay_ids                  : list
    """
    n_total     = len(encoded_data['states'])
    action_low  = params.get('action_low',  [0.0] * params.get('action_dim', 2))
    action_high = params.get('action_high', [1.0] * params.get('action_dim', 2))
    alphas_f    = [float(a) for a in alphas]
    num_tau     = params.get('bnd_num_tau', params.get('num_iqn_samples_est', 64))

    n_valid = sum(
        1 for i in range(n_total)
        if float(encoded_data['rewards'][i][int(encoded_data['lengths'][i].flat[0]) - 1].flat[-1])
        in (-1.0, 1.0)
    )
    print(
        f"Grid CVaR evaluation (M={M}×{M}, {len(alphas_f)} alpha levels, "
        f"max over grid) — {n_valid} valid trajectories...",
        flush=True,
    )

    ns_bucket = {'v_dn': [], 'v_rn': [], 'stay_ids': []}
    s_bucket  = {'v_dn': [], 'v_rn': [], 'stay_ids': []}
    n_done = 0

    for traj in range(n_total):
        traj_len   = int(encoded_data['lengths'][traj].flat[0])
        traj_r     = encoded_data['rewards'][traj][:traj_len]
        terminal_r = float(traj_r[-1].flat[-1])
        if terminal_r not in (-1.0, 1.0):
            continue

        category = -1 if terminal_r < 0 else 1
        traj_sid = encoded_data['stay_ids'][traj]
        states   = encoded_data['states'][traj][:traj_len]  # (T, state_dim)

        vdn = grid_cvar_batch(
            states, qnet_dn, alphas_f, M=M,
            action_low=action_low, action_high=action_high,
            num_tau=num_tau, agg='max',
        )  # (T, n_alphas)
        vdn = np.clip(vdn, -1.0, 0.0)

        vrn = grid_cvar_batch(
            states, qnet_rn, alphas_f, M=M,
            action_low=action_low, action_high=action_high,
            num_tau=num_tau, agg='max',
        )  # (T, n_alphas)
        vrn = np.clip(vrn, 0.0, 1.0)

        bucket = ns_bucket if category == -1 else s_bucket
        bucket['v_dn'].append(vdn)
        bucket['v_rn'].append(vrn)
        bucket['stay_ids'].append(traj_sid)

        n_done += 1
        if log_every_traj and (n_done % log_every_traj == 0):
            print(f"  {n_done}/{n_valid} trajectories processed", flush=True)

    def _pack(bucket):
        return {
            'dn_q_selected_action_traj': bucket['v_dn'],
            'rn_q_selected_action_traj': bucket['v_rn'],
            'dn_v_median_traj':          bucket['v_dn'],
            'rn_v_median_traj':          bucket['v_rn'],
            'stay_ids':                  bucket['stay_ids'],
        }

    return {'nonsurvivors': _pack(ns_bucket), 'survivors': _pack(s_bucket)}


#############################
##     MAIN EXECUTOR
#############################

@click.command()
@click.option('--config', '-c', default='dqn_baseline')
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
@click.option('--data', '-d', default='test')
@click.option('--plot_hists', is_flag=True, help="produce histograms of the computed values of the specified dataset")
def run(config, options, data, plot_hists):
    """Using the specified configuration, evaluate the trained D and R networks with the specified dataset."""

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, f'configs/{config}.yaml')))

    split_fname = params['checkpoint_fname'].split("/")

    addon = "_overlap" if data=='overlap' else ""

    local_storage_dir = os.path.join("stats", "/".join(split_fname[-3:]))
    os.makedirs(local_storage_dir, exist_ok=True)

    # Replace configuration parameters by command line provided 'options'
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt
    print('Parameters')
    for key in params:
        print(key, params[key])
    print('=' * 30)

    # Set the seeds
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    rng = np.random.RandomState(params['seed'])

    # Load the data
    encoded_data = np.load(os.path.join(params['data_dir'], f'encoded_{data}.npz'), allow_pickle=True)

    data_input_dim = encoded_data['states'].shape[-1]
    ckpt_path = os.path.join(params['checkpoint_fname'], "best_q_parametersnegative.pt")
    ckpt_input_dim = _get_checkpoint_input_dim(ckpt_path, device)

    # For ContinuousIQN, the checkpoint head takes [state ∥ action] so head_dim = state_dim + action_dim.
    # Recover the pure state_dim before comparing with the encoded data dimension.
    ckpt_state_dim = ckpt_input_dim
    if ckpt_input_dim is not None and params.get('model') == 'ContinuousIQN':
        ckpt_state_dim = ckpt_input_dim - params.get('action_dim', 0)

    if ckpt_state_dim is not None and ckpt_state_dim != data_input_dim:
        print(
            "Input-dimension mismatch between encoded data and checkpoint. "
            f"Encoded data has {data_input_dim} features, but checkpoint expects {ckpt_state_dim}.",
            flush=True,
        )
        encoded_data = _coerce_encoded_state_dim(
            encoded_data,
            ckpt_state_dim,
            label=f"encoded_{data}",
        )
        data_input_dim = ckpt_state_dim

    params["input_dim"] = data_input_dim

    # For ContinuousIQN, compute per-dim state mean/std from training data so that
    # the network's _s_mean/_s_scale are 116-dim (matching the NCDE-encoded states)
    # rather than falling back to the 4-dim SpaceEnv hardcoded defaults.
    if params.get('model') == 'ContinuousIQN':
        print("Computing state normalisation statistics from training data...")
        tr_npz  = np.load(os.path.join(params['data_dir'], 'encoded_train.npz'), allow_pickle=True)
        val_npz = np.load(os.path.join(params['data_dir'], 'encoded_validation.npz'), allow_pickle=True)
        state_dim = params["input_dim"]
        tr_states = _adjust_state_dim(tr_npz['states'], state_dim, label="encoded_train")
        val_states = _adjust_state_dim(val_npz['states'], state_dim, label="encoded_validation")
        all_states = np.concatenate([
            tr_states.reshape(-1, state_dim),
            val_states.reshape(-1, state_dim),
        ], axis=0)
        params['state_mean'] = all_states.mean(axis=0)
        params['state_std']  = all_states.std(axis=0).clip(min=1e-8)

    print("Loading best Q-Networks and making Q-values ...")

    # Initialize and load the trained models
    qnet_dn = load_best_rl(params, "negative", device)
    qnet_rn = load_best_rl(params, "positive", device)

    continuous = params['model'] == 'ContinuousIQN'
    distributional = False  # overridden below for discrete IQN
    VaR_thresholds = np.round(np.linspace(0.05, 1.0, num=20), decimals=2)

    if continuous:
        alpha_sweep = params.get('bnd_alpha_sweep', VaR_thresholds)
        alpha_sweep = np.asarray(alpha_sweep, dtype=np.float32)
        alphas_f    = [float(a) for a in alpha_sweep]
        bnd_method  = params.get('bnd_method', 'predictor_corrector')

        if bnd_method == 'predictor_corrector':
            pc_kwargs = dict(
                bnd_M=params.get('bnd_M', 5),
                bnd_h0=params.get('bnd_h0', 0.05),
                bnd_eps_tol=params.get('bnd_eps_tol', 1e-4),
                bnd_eps_close=params.get('bnd_eps_close', 0.02),
                bnd_eps_dup=params.get('bnd_eps_dup', 0.02),
                bnd_eta=params.get('bnd_eta', 1.5),
                bnd_C_max=params.get('bnd_C_max', 10),
                bnd_num_tau=params.get('bnd_num_tau', params.get('num_iqn_samples_est', 64)),
                log_every_traj=params.get('eval_log_every_traj', 10),
            )

            # --- R-net: single calibration (not swept — sweeping both D and R
            #     would require N_D × N_R full passes; the D-net provides the
            #     primary dead-end signal so the sweep is concentrated there).
            if params.get('bnd_delta_D_auto', True):
                delta_D_rn = _probe_q_percentile(
                    qnet_rn, encoded_data, params,
                    percentile=params.get('bnd_delta_R_percentile', 75),
                    n_samples=500, alpha=float(alphas_f[0]),
                )
            else:
                delta_D_rn = params.get('bnd_delta_R', 0.5)

            data_by_alpha_rn = get_continuous_dead_end_data(
                qnet_rn, encoded_data, device, params,
                bnd_alphas=alphas_f,
                bnd_delta_D=delta_D_rn,
                checkpoint_path=os.path.join(params['checkpoint_fname'], f'pc_rn_ckpt{addon}.pkl'),
                **pc_kwargs,
            )

            # --- D-net: pre-screen candidates with a fast grid scan, then run PC once.
            # Running the full predictor-corrector for each candidate is O(N_cand × n_traj × T × N_PC_iters).
            # The grid scan is O(N_cand × n_traj) batched forward passes — typically 100-1000× cheaper.
            # The R-net grid pass is cached once since delta_D_rn is fixed across candidates.
            sweep_percentiles = params.get('bnd_delta_D_sweep_percentiles', [10, 25, 40, 50, 60, 75])

            def _merge_results(data_dn, data_rn, alphas_f):
                merged = {}
                for alpha in alphas_f:
                    cols_dn = ['traj', 'step', 'f_D', 'v_dn', 'category', 'stay_id']
                    df_dn   = data_dn[alpha][cols_dn].copy()
                    df_rn   = data_rn[alpha][['traj', 'step', 'f_D']].rename(columns={'f_D': 'f_R'})
                    m       = df_dn.merge(df_rn, on=['traj', 'step'])
                    m['v_rn'] = 1.0 - m['f_R']
                    merged[alpha] = m
                return merged

            if not params.get('bnd_delta_D_auto', True):
                delta_D_dn = params.get('bnd_delta_D', -0.5)
                pct_label  = 'fixed'
            else:
                candidates = _probe_q_percentiles(
                    qnet_dn, encoded_data, params,
                    percentiles=sweep_percentiles, n_samples=500, alpha=float(alphas_f[0]),
                )
                delta_D_dn, pct_label = _select_delta_D_grid(
                    qnet_dn, qnet_rn, encoded_data, params,
                    candidates=candidates, alphas_f=alphas_f,
                    delta_D_rn=delta_D_rn,
                    M=params.get('bnd_M', 5),
                )

            print(f"\nRunning predictor-corrector with delta_D_dn={delta_D_dn:.4f} ({pct_label})...", flush=True)
            data_by_alpha_dn = get_continuous_dead_end_data(
                qnet_dn, encoded_data, device, params,
                bnd_alphas=alphas_f,
                bnd_delta_D=delta_D_dn,
                checkpoint_path=os.path.join(
                    params['checkpoint_fname'], f'pc_dn_{pct_label}_ckpt{addon}.pkl'
                ),
                **pc_kwargs,
            )

            results = _assemble_pc_results(
                _merge_results(data_by_alpha_dn, data_by_alpha_rn, alphas_f), alphas_f
            )
            final_auc = _quick_auc(results)
            print(f"PC AUC with delta_D {pct_label}={delta_D_dn:.4f}: {final_auc:.4f}", flush=True)

            value_data = {
                'alpha':      alpha_sweep,
                'delta_D_dn': delta_D_dn,
                'delta_D_rn': delta_D_rn,
                'results':    results,
            }

        else:  # 'grid_cvar' — fast approximation, no delta_D needed
            results = get_continuous_dead_end_data_grid_cvar(
                qnet_dn, qnet_rn, encoded_data, device, params,
                alphas=alphas_f,
                M=params.get('bnd_M', 10),
                log_every_traj=params.get('eval_log_every_traj', 10),
            )
            value_data = {'alpha': alpha_sweep, 'results': results}
    else:
        # Discrete action space: retrieve Q-values for each state (for all actions)
        distributional = params['model'] == "IQN"
        data = get_dn_rn_info(qnet_dn, qnet_rn, encoded_data, device, distributional=distributional)
        data = pd.DataFrame(data)

        results = {"survivors": {}, "nonsurvivors": {}}
        non_survivor_trajectories = sorted(data[data.category == -1].traj.unique().tolist())
        survivor_trajectories = sorted(data[data.category == 1].traj.unique().tolist())
        for i, trajectories in enumerate([non_survivor_trajectories, survivor_trajectories]):
            if i == 0:
                traj_type = "nonsurvivors"
                print("----------- Non-survivors")
            else:
                traj_type = "survivors"
                print("+++++++++++ Survivors")

            dn_q_selected_action_traj = []
            rn_q_selected_action_traj = []
            dn_v_median_traj = []
            rn_v_median_traj = []
            sid_traj = []
            for traj in trajectories:
                d = data[data.traj == traj]
                sid_traj.append(d.stay_id.tolist()[0])

                dn_q_traj = np.array(d.q_dn.tolist(), dtype=np.float32)
                rn_q_traj = np.array(d.q_rn.tolist(), dtype=np.float32)
                if not distributional:
                    dn_q_selected_action = [d.q_dn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_dn.shape[0])]
                    rn_q_selected_action = [d.q_rn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_rn.shape[0])]
                    dn_q_selected_action_traj.append(dn_q_selected_action)
                    rn_q_selected_action_traj.append(rn_q_selected_action)
                    dn_v_median_traj.append(np.median(dn_q_traj, axis=1))
                    rn_v_median_traj.append(np.median(rn_q_traj, axis=1))
                else:
                    (num_steps, num_samples, num_actions) = dn_q_traj.shape
                    cvar_dn_traj = np.zeros((num_steps, len(VaR_thresholds), num_actions))
                    cvar_rn_traj = np.zeros((num_steps, len(VaR_thresholds), num_actions))
                    for ii, var in enumerate(VaR_thresholds):
                        var_cutoff = round(var * num_samples)
                        cvar_dn_traj[:, ii, :] = np.mean(dn_q_traj[:, :var_cutoff, :], axis=1)
                        cvar_rn_traj[:, ii, :] = np.mean(rn_q_traj[:, :var_cutoff, :], axis=1)

                    dn_q_selected_action = torch.from_numpy(cvar_dn_traj).gather(2, torch.from_numpy(d.a.values).unsqueeze(-1).unsqueeze(-1).expand(num_steps, 20, 1)).squeeze().numpy()
                    rn_q_selected_action = torch.from_numpy(cvar_rn_traj).gather(2, torch.from_numpy(d.a.values).unsqueeze(-1).unsqueeze(-1).expand(num_steps, 20, 1)).squeeze().numpy()
                    dn_q_selected_action_traj.append(dn_q_selected_action)
                    rn_q_selected_action_traj.append(rn_q_selected_action)
                    dn_v_median_traj.append(np.median(cvar_dn_traj, axis=2))
                    rn_v_median_traj.append(np.median(cvar_rn_traj, axis=2))

            results[traj_type]["dn_q_selected_action_traj"] = dn_q_selected_action_traj
            results[traj_type]["rn_q_selected_action_traj"] = rn_q_selected_action_traj
            results[traj_type]["stay_ids"] = sid_traj
            results[traj_type]["dn_v_median_traj"] = dn_v_median_traj
            results[traj_type]["rn_v_median_traj"] = rn_v_median_traj

        value_data = data

    print("Writing value_data...", flush=True)
    with open(os.path.join(params['checkpoint_fname'], "value_data"+addon+".pkl"), "wb") as f:
        pickle.dump(value_data, f)
    with open(os.path.join(local_storage_dir, "value_data"+addon+".pkl"), "wb") as f:
        pickle.dump(value_data, f)

    print("Writing pre_flag_results...", flush=True)
    with open(os.path.join(params['checkpoint_fname'], "pre_flag_results"+addon+".pkl"), "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(local_storage_dir, "pre_flag_results"+addon+".pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Done — pre_flag_results written successfully.", flush=True)


    if plot_hists:
        # Need to reconvert the  results dictionary back into a dataframe and expand back out to a trajectory based dataframe (similar to `data`)
        surv_df, nonsurv_df = create_analysis_df(results, len(survivor_trajectories), len(non_survivor_trajectories))

        # Create arrays of additional subsampling of the computed values
        # This accounts for the extra outer loop needed to analyze the distributional RL results
        if continuous:
            alpha_sweep = params.get('bnd_alpha_sweep', VaR_thresholds)
            alpha_sweep = np.asarray(alpha_sweep, dtype=np.float32)
            if len(alpha_sweep) > 1:
                VaR_thresholds = alpha_sweep
            else:
                VaR_thresholds = [None]
        elif not distributional:
            VaR_thresholds = [None]
        else:
            VaR_thresholds =  np.round(np.linspace(0.05, 1.0, num=20), decimals=2)

        time_steps = [-72, -48, -24, -12, -8, -4, -1]  # The hours before the terminal observation that we want to construct histograms for

        for ivar, var in enumerate(VaR_thresholds):
            # Adjust ivar for non-distributional results
            if var is None:
                ivar = None
                append = ''
            else:
                append = f'_var_{var}'
                
            figname = "computed_value_histograms" + append + addon + '.png'
            
            # Initialize figure frame to place the histograms in
            fig, full_axs = plt.subplots(4, len(time_steps), figsize=(24, 6), dpi=800, sharey=True)
            plots = {'fig': fig, 'axs': full_axs}

            for i, step_num in enumerate(time_steps):
                axs = plots['axs'][:, i]
                plot_value_hists(axs, nonsurv_df, surv_df, step_num, var_idx=ivar)
            
            plots['fig'].tight_layout()
            sns.despine(plots['fig'])
            plots['fig'].savefig(os.path.join(params['checkpoint_fname'], figname))
            plt.close("all")




if __name__ == '__main__':
    run()