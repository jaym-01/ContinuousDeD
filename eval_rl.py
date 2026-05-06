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

from rl_utils import RLDataLoader, DQN_Agent, IQN_Agent, create_analysis_df, ContinuousIQN_OfflineAgent
from plot_utils import plot_value_hists
from analysis_utils import get_dn_rn_info, create_analysis_df
from boundary_tracing import (dead_end_volume_fraction,
                               dead_end_volume_fraction_multi_alpha,
                               dead_end_volume_fraction_grid,
                               dead_end_volume_fraction_grid_batch)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
##     HELPER FUNCTIONS
#############################

def load_best_rl(params, sided_Q, device):
    best_rl_path = os.path.join(params['checkpoint_fname'], f"best_q_parameters{sided_Q}.pt")
    best_rl_checkpoint = torch.load(best_rl_path, weights_only=False)

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


def get_continuous_dead_end_data(qnet_dn, encoded_data, device, params,
                                  bnd_M=10, bnd_alphas=(0.1,), bnd_delta_D=-0.5):
    """Compute dead-end volume fraction f_D per state for all alpha values.

    Processes one trajectory at a time with a single batched forward pass of
    size T×M² (T = trajectory length), so the GPU sees a meaningful batch rather
    than one state at a time.

    Returns
    -------
    data_by_alpha : dict {alpha: pd.DataFrame}
    """
    n_total = len(encoded_data['states'])
    action_low  = params.get('action_low',  [0.0] * params.get('action_dim', 2))
    action_high = params.get('action_high', [1.0] * params.get('action_dim', 2))
    alphas = list(bnd_alphas)
    alphas_f = [float(a) for a in alphas]

    rows_by_alpha = {a: [] for a in alphas_f}

    n_valid = sum(
        1 for i in range(n_total)
        if float(encoded_data['rewards'][i][int(encoded_data['lengths'][i].flat[0]) - 1].flat[-1]) in (-1.0, 1.0)
    )
    print(f"Computing dead-end volume fractions via grid estimator "
          f"(M={bnd_M}, {len(alphas)} alpha levels, 1 forward pass per trajectory) "
          f"— {n_valid} valid trajectories...")

    n_done = 0
    for traj in range(n_total):
        traj_len   = int(encoded_data['lengths'][traj].flat[0])
        traj_r     = encoded_data['rewards'][traj][:traj_len]
        terminal_r = float(traj_r[-1].flat[-1])
        if terminal_r not in (-1.0, 1.0):
            continue

        category = -1 if terminal_r < 0 else 1
        traj_sid = encoded_data['stay_ids'][traj]
        traj_a   = encoded_data['actions'][traj][:traj_len]
        states   = encoded_data['states'][traj][:traj_len]  # (T, state_dim)

        # One forward pass for all T states × M² actions
        f_D_mat = dead_end_volume_fraction_grid_batch(
            states, qnet_dn,
            alphas=alphas_f,
            delta_D=bnd_delta_D,
            M=bnd_M,
            action_low=action_low,
            action_high=action_high,
        )  # (T, len(alphas))

        for t in range(traj_len):
            for j, alpha in enumerate(alphas_f):
                f_D = float(f_D_mat[t, j])
                rows_by_alpha[alpha].append({
                    'traj':     traj,
                    'step':     t,
                    's':        states[t],
                    'a':        traj_a[t],
                    'f_D':      f_D,
                    'v_dn':     -f_D,
                    'v_rn':     1.0 - f_D,
                    'category': category,
                    'stay_id':  traj_sid,
                })

        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{n_valid} trajectories processed")

    return {a: pd.DataFrame(rows) for a, rows in rows_by_alpha.items()}

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
    if not os.path.exists(local_storage_dir):
        os.makedirs(local_storage_dir)

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

    params["input_dim"] = encoded_data['states'].shape[-1]

    # For ContinuousIQN, compute per-dim state mean/std from training data so that
    # the network's _s_mean/_s_scale are 116-dim (matching the NCDE-encoded states)
    # rather than falling back to the 4-dim SpaceEnv hardcoded defaults.
    if params.get('model') == 'ContinuousIQN':
        print("Computing state normalisation statistics from training data...")
        tr_npz  = np.load(os.path.join(params['data_dir'], 'encoded_train.npz'), allow_pickle=True)
        val_npz = np.load(os.path.join(params['data_dir'], 'encoded_validation.npz'), allow_pickle=True)
        state_dim = params["input_dim"]
        all_states = np.concatenate([
            tr_npz['states'].reshape(-1, state_dim),
            val_npz['states'].reshape(-1, state_dim),
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
        # Process all alpha levels in one pass per state (shared grid forward pass)
        data_by_alpha = get_continuous_dead_end_data(
            qnet_dn, encoded_data, device, params,
            bnd_M=params.get('bnd_M', 10),
            bnd_alphas=[float(a) for a in alpha_sweep],
            bnd_delta_D=params.get('bnd_delta_D', -0.5),
        )

        first_alpha = float(alpha_sweep[0])
        data_ref = data_by_alpha[first_alpha]
        results = {"survivors": {}, "nonsurvivors": {}}
        non_survivor_trajectories = sorted(data_ref[data_ref.category == -1].traj.unique().tolist())
        survivor_trajectories = sorted(data_ref[data_ref.category == 1].traj.unique().tolist())
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
                d_ref = data_ref[data_ref.traj == traj]
                sid_traj.append(d_ref.stay_id.tolist()[0])

                f_D_alpha = []
                for alpha in alpha_sweep:
                    d_alpha = data_by_alpha[float(alpha)]
                    d_traj = d_alpha[d_alpha.traj == traj]
                    f_D_alpha.append(d_traj.f_D.values.astype(np.float32))

                f_D_stack = np.stack(f_D_alpha, axis=1)
                dn_q_selected_action_traj.append(-f_D_stack)
                rn_q_selected_action_traj.append(1.0 - f_D_stack)
                dn_v_median_traj.append(-f_D_stack)
                rn_v_median_traj.append(1.0 - f_D_stack)

            results[traj_type]["dn_q_selected_action_traj"] = dn_q_selected_action_traj
            results[traj_type]["rn_q_selected_action_traj"] = rn_q_selected_action_traj
            results[traj_type]["stay_ids"] = sid_traj
            results[traj_type]["dn_v_median_traj"] = dn_v_median_traj
            results[traj_type]["rn_v_median_traj"] = rn_v_median_traj

        value_data = {"alpha": alpha_sweep, "data": data_by_alpha}
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

    with open(os.path.join(params['checkpoint_fname'], "value_data"+addon+".pkl"), "wb") as f:
        pickle.dump(value_data, f)

    with open(os.path.join(local_storage_dir, "value_data"+addon+".pkl"), "wb") as f:
        pickle.dump(value_data, f)

    with open(os.path.join(params['checkpoint_fname'], "pre_flag_results"+addon+".pkl"), "wb") as f:
        pickle.dump(results, f)

    print("writing results")
    with open(os.path.join(local_storage_dir, "pre_flag_results"+addon+".pkl"), "wb") as f:
        pickle.dump(results, f)


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