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

from rl_utils import RLDataLoader, DQN_Agent, IQN_Agent, ValueFlows_Agent, C51_Agent, C51_2bin_Agent
from plot_utils import plot_value_hists
from analysis_utils import get_dn_rn_info, create_analysis_df

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
##     HELPER FUNCTIONS
#############################

def load_best_rl(params, sided_Q, device):
    best_rl_path = os.path.join(params['checkpoint_fname'], f"best_q_parameters{sided_Q}.pt")
    best_rl_checkpoint = torch.load(best_rl_path, map_location='cpu', weights_only=False)
    
    # Initialize the model
    if params['model'] == 'DQN':
        model = DQN_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    elif params['model'] == 'IQN':
        model = IQN_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    elif params['model'] == 'ValueFlows':
        model = ValueFlows_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    elif params['model'] == 'C51':
        model = C51_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    elif params['model'] == 'C51_2bin':
        model = C51_2bin_Agent(params['input_dim'], params, sided_Q=sided_Q, device=device)
    else:
        raise NotImplementedError('The provided model type has not yet been defined, please use DQN, IQN, ValueFlows, C51, or C51_2bin')

    # Load the best performing parameters (based on validation loss) into the model
    # C51 checkpoints saved before 'support' was moved from C51Network to C51_Agent
    # may contain a 'support' key that no longer belongs in the network state dict.
    network_state = best_rl_checkpoint['rl_network_state_dict']
    network_state = {k: v for k, v in network_state.items() if k != 'support'}
    model.network.load_state_dict(network_state)
    model.eval()
    print(f"{sided_Q.capitalize()} Q-Network loaded")
    return model

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

    split = data  # rename to avoid shadowing when data is reassigned below

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, f'configs/{config}.yaml')))

    split_fname = params['checkpoint_fname'].split("/")

    addon = "_overlap" if split=='overlap' else ""

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
    encoded_data = np.load(os.path.join(params['data_dir'], f'encoded_{split}.npz'), allow_pickle=True)

    params["input_dim"] = encoded_data['states'].shape[-1]

    print("Loading best Q-Networks and making Q-values ...")
    
    # Initialize and load the trained models
    qnet_dn = load_best_rl(params, "negative", device)
    qnet_rn = load_best_rl(params, "positive", device)

    # Retrieve the Q-values for each state (for all actions)
    distributional = params['model'] in ("IQN", "ValueFlows", "C51", "C51_2bin")
    data = get_dn_rn_info(qnet_dn, qnet_rn, encoded_data, device, distributional=distributional)

    with open(os.path.join(params['checkpoint_fname'], "value_data"+addon+".pkl"), "wb") as f:
        pickle.dump(data, f)

    with open(os.path.join(local_storage_dir, "value_data"+addon+".pkl"), "wb") as f:
        pickle.dump(data, f)

    data = pd.DataFrame(data)  # convert dict returned by get_dn_rn_info to DataFrame

    VaR_thresholds =  np.round(np.linspace(0.05, 1.0, num=20), decimals=2)
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

        dn_q_traj = []
        rn_q_traj = []
        dn_q_selected_action_traj = []
        rn_q_selected_action_traj = []
        dn_q_CVaR_traj = []
        rn_q_CVaR_traj = []
        sid_traj = []  # Keep track of the specific stay IDs associated with each patient trajectory (for analysis purposes)
        for traj in trajectories:
            d = data[data.traj == traj]
            sid_traj.append(d.stay_id.tolist()[0])  # We only need one Stay ID per trajectory (was repeated across time for each trajectory in this column)
            dn_q_traj.append(np.array(d.q_dn.tolist(), dtype=np.float32))
            rn_q_traj.append(np.array(d.q_rn.tolist(), dtype=np.float32))
            if not distributional:
                dn_q_selected_action = [d.q_dn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_dn.shape[0])] 
                rn_q_selected_action = [d.q_rn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_rn.shape[0])] 
            else:
                (num_steps, num_samples, num_actions) = dn_q_traj[-1].shape
                cvar_dn_traj = np.zeros((num_steps, len(VaR_thresholds), num_actions))
                cvar_rn_traj = np.zeros((num_steps, len(VaR_thresholds), num_actions))
                # Loop through all VaR thresholds and the extract the selected action
                for ivar, var in enumerate(VaR_thresholds):
                    var_cutoff = round(var*num_samples)
                    cvar_dn_traj[:, ivar, :] = np.mean(dn_q_traj[-1][:, :var_cutoff, :], axis=1)
                    cvar_rn_traj[:, ivar, :] = np.mean(rn_q_traj[-1][:, :var_cutoff, :], axis=1)

                # Select the CVaR values for only the actions that were actually used
                dn_q_selected_action = torch.from_numpy(cvar_dn_traj).gather(2, torch.from_numpy(d.a.values).unsqueeze(-1).unsqueeze(-1).expand(num_steps, 20, 1)).squeeze().numpy()
                rn_q_selected_action = torch.from_numpy(cvar_rn_traj).gather(2, torch.from_numpy(d.a.values).unsqueeze(-1).unsqueeze(-1).expand(num_steps, 20, 1)).squeeze().numpy()

                # Record the CVaR values for all VaR thresholds (for computing the median across actions)
                dn_q_CVaR_traj.append(cvar_dn_traj)
                rn_q_CVaR_traj.append(cvar_rn_traj)
                
            dn_q_selected_action_traj.append(dn_q_selected_action)
            rn_q_selected_action_traj.append(rn_q_selected_action)
        

        results[traj_type]["dn_q_selected_action_traj"] = dn_q_selected_action_traj
        results[traj_type]["rn_q_selected_action_traj"] = rn_q_selected_action_traj
        results[traj_type]["stay_ids"] = sid_traj
        if not distributional:
            results[traj_type]["dn_v_median_traj"] = [np.median(q, axis=1) for q in dn_q_traj]
            results[traj_type]["rn_v_median_traj"] = [np.median(q, axis=1) for q in rn_q_traj]
        else:
            results[traj_type]["dn_v_median_traj"] = [np.median(q, axis=2) for q in dn_q_CVaR_traj]
            results[traj_type]["rn_v_median_traj"] = [np.median(q, axis=2) for q in rn_q_CVaR_traj]

    with open(os.path.join(params['checkpoint_fname'], "pre_flag_results"+addon+".pkl"), "wb") as f:
        pickle.dump(results, f)

    with open(os.path.join(local_storage_dir, "pre_flag_results"+addon+".pkl"), "wb") as f:
        pickle.dump(results, f)


    if plot_hists:
        # Need to reconvert the  results dictionary back into a dataframe and expand back out to a trajectory based dataframe (similar to `data`)
        surv_df, nonsurv_df = create_analysis_df(results, len(survivor_trajectories), len(non_survivor_trajectories))

        # Create arrays of additional subsampling of the computed values
        # This accounts for the extra outer loop needed to analyze the distributional RL results
        if distributional: 
            VaR_thresholds =  np.round(np.linspace(0.05, 1.0, num=20), decimals=2)
        else:
            VaR_thresholds = [None]

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