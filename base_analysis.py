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

from analysis_utils import create_analysis_df
from plot_utils import plot_value_hists  # Perhaps wrap this value estimate plotting into plot_utils?
from thresholds import th

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

#############################
##     HELPER FUNCTIONS
#############################


# Flag comparison functions (if estimated value drops below )
def compare_dist(row):
    return np.logical_and(row["v_dn"]<th.new_dn_yel, row["v_rn"]<th.new_rn_yel)

def compare_ded(row):
    return np.logical_and(row["v_dn"]<th.ded_dn_yel, row["v_rn"]<th.ded_rn_yel)

def flag_in_range(flag_time, traj_length, window_pre, window_post):
    """Determine if flag occurs within the range"""
    # If flag is too early
    if flag_time < window_pre:  
        return False
    # If flag is too late
    if traj_length - flag_time <= window_post:
        return False
    
    return True


def comp_flag_agg_values(df, window_pre=12, window_post=8, distributional=False, same_thresh=False):
    """
    Aggregate values across the test set within a specified window of the first observed flag.

    We want to investigate the overall trends of the DeD computed values among the patient
    trajectories contained in the test set. This requires that we align the trajectories. The 
    way that we've chosen to do this (following what was done in the original DeD paper) is
    to pick a window of time before and after the first computed flag, aligning each trajectory
    by this first flag and then aggregating over the number of hours before and afterwards. We 
    discard all trajectories that either:
     - are too short for the specified window [-before, after]
     - the flag occurs too early or too late in the trajectory such that the desired [-before, after]
        window does not fall within the recorded observations

    Args:
        df (pd.DataFrame): A dataframe produced by `create_analysis_df`, will contain only trajectories
            for either surviving or nonsurviving patients.
        window_pre (int): the minimum window of time (hours) prior to the first flag over which we aggregate computed values
        window_post (int): the minimum window of time (hours) after the first flag over which we aggregate computed values
        distributional (bool, default=False): Whether we have a set of values (for each setting of the VaR threshold used),
            if True, an additional loop is run over all these settings for which separate columns indicating the alpha value
            for the trajectory is included in the output dataframe.

    Returns:
        monitor (pd.DataFrame): A dataframe containing the outputs of the aggregated value estimates.
    """

    VaR_thresholds = np.round(np.linspace(0.05, 1.0, num=20), decimals=2)

    # Initialize the output dataframe
    monitor = pd.DataFrame()

    # Loop through each trajectory in df
    for traj in df.traj.unique():
        dt = df[df.traj==traj]
        if len(dt) < window_pre+window_post:
            continue
        # Calculate flags
        if distributional:
            if same_thresh:
                flags = np.vstack(dt.apply(compare_ded, axis=1).values)
            else:
                flags = np.vstack(dt.apply(compare_dist, axis=1).values)
            # Loop through the various VaR thresholds (e.g. the computed values are the CVaR of the VaR alpha quantile)
            for i, var in enumerate(VaR_thresholds):
                var_flags = flags[:, i]
                if np.any(var_flags):
                    # Extract the first flag
                    first_flag = np.where(var_flags == True)[0][0]
                    # Check whether there is enough time before and after this first flag
                    if not flag_in_range(first_flag, len(dt), window_pre, window_post):
                        continue
                    # Extract the subsection of the trajectory "centered" around this first flag    
                    temp_df = dt.iloc[(first_flag-window_pre):(first_flag+window_post)].copy()
                    
                    # Extract only the values associated with this VaR Threshold
                    for item in temp_df.columns: 
                        if item in ['traj', 'step']:
                            continue
                        temp_df[item] = np.vstack(temp_df[item].values)[:, i]
                    temp_df['step'] = np.arange(-window_pre, len(temp_df)-window_pre)
                    temp_df['var_alpha'] = var
                    temp_df['var_index'] = i

                    # Add the trajectory subsection to the output DataFrame
                    monitor = pd.concat((monitor, temp_df))
        else:
            flags = np.vstack(dt.apply(compare_ded, axis=1).values)
            if np.any(flags):
                # Extract the first flag
                first_flag = np.where(flags==True)[0][0]
                # Check whether there is enought time before and after this first flag
                if not flag_in_range(first_flag, len(dt), window_pre, window_post):
                    continue
                # Extract the subsection of the trajectory "centered" around this first flag
                temp_df = dt.iloc[(first_flag-window_pre):(first_flag+window_post)].copy()
                temp_df['step'] = np.arange(-window_pre, len(temp_df)-window_pre)

                # Add the trajectory subsection to the output DataFrame
                monitor = pd.concat((monitor, temp_df))

    return monitor        


#############################
##     MAIN EXECUTOR
#############################

@click.command()
@click.option('--analysis', '-a', multiple=True, default=['agg_values'])
def run(analysis):
    """Aggregate results from model evaluation to get overall value traces..."""

    window_pre = 12
    window_post = 8

    # Load the data
    metrics_dir = '/ais/bulbasaur/twkillian/UncDeD_Results/'
    # Baseline DDQN results
    with open(os.path.join(metrics_dir, "DQN", "dqn_baseline", "pre_flag_results.pkl"), "rb") as f:
        dqn_results = pickle.load(f)
    
    # DDQN + CQL Results (using smallest CQL weight)
    with open(os.path.join(metrics_dir, "DQN", "dqn_cql_wt_p05", "pre_flag_results.pkl"), "rb") as f:
        dqn_cql_results = pickle.load(f)

    # IQN Baseline results
    with open(os.path.join(metrics_dir, "IQN", "iqn_baseline", "pre_flag_results.pkl"), "rb") as f: 
       iqn_results = pickle.load(f)

    # IQN + CQL Results (using smallest CQL weight)
    with open(os.path.join(metrics_dir, "IQN", "iqn_cql_wt_p05", "pre_flag_results.pkl"), "rb") as f:
        iqn_cql_results = pickle.load(f)

    # Aggregate the computed results for analysis
    num_nonsurvivors = len(dqn_results['nonsurvivors']['dn_q_selected_action_traj'])
    num_survivors = len(dqn_results['survivors']['dn_q_selected_action_traj'])
    
    # DQN
    dqn_surv_df, dqn_nonsurv_df = create_analysis_df(dqn_results, num_survivors, num_nonsurvivors)
    # DQN + CQL
    dqn_cql_surv_df, dqn_cql_nonsurv_df = create_analysis_df(dqn_cql_results, num_survivors, num_nonsurvivors)
    # IQN
    iqn_surv_df, iqn_nonsurv_df = create_analysis_df(iqn_results, num_survivors, num_nonsurvivors)
    # IQN + CQL
    iqn_cql_surv_df, iqn_cql_nonsurv_df = create_analysis_df(iqn_cql_results, num_survivors, num_nonsurvivors)

    if 'agg_values' in analysis:
        ## Calculate the flags (all with the same flags, then also with separate flags for the distributional/CQL approaches...)
        # DQN
        dqn_s_vals = comp_flag_agg_values(dqn_surv_df)
        dqn_ns_vals = comp_flag_agg_values(dqn_nonsurv_df)
        # DQN + CQL
        dqn_cql_s_vals = comp_flag_agg_values(dqn_cql_surv_df)
        dqn_cql_ns_vals = comp_flag_agg_values(dqn_cql_nonsurv_df)
        # IQN
        iqn_s_vals = comp_flag_agg_values(iqn_surv_df, distributional=True)
        iqn_ns_vals = comp_flag_agg_values(iqn_nonsurv_df, distributional=True)
        # IQN + CQL
        iqn_cql_s_vals = comp_flag_agg_values(iqn_cql_surv_df, distributional=True)
        iqn_cql_ns_vals = comp_flag_agg_values(iqn_cql_nonsurv_df, distributional=True)

        ## Plot all value traces over a relatively small window before and after the flag is raised (12h, 8h)?
          # We do this twice (once for IQN and once for IQN+CQL)

        ns_colors = sns.color_palette("Blues", n_colors=20)
        s_colors = sns.color_palette("Greens", n_colors=20)
        titles = [r"$Q_{D}$", r"$Q_{R}$", r"$V_{D}$", r"$V_{R}$"]

        # First for IQN
        fig, axs = plt.subplots(2,2, figsize=(12,8), dpi=600, sharex=True)
        
        for i, item in enumerate(['q_dn', 'q_rn', 'v_dn', 'v_rn']):
            ax = axs[i//2, i%2]
            # Extract the DQN results
            dqn_ns_item = dqn_ns_vals[item].values
            dqn_ns_steps = dqn_ns_vals['step'].values
            dqn_s_item = dqn_s_vals[item].values
            dqn_s_steps = dqn_s_vals['step'].values

            # Extract the DQN+CQL results
            dqn_cql_ns_item = dqn_cql_ns_vals[item].values
            dqn_cql_ns_steps = dqn_cql_ns_vals['step'].values
            dqn_cql_s_item = dqn_cql_s_vals[item].values
            dqn_cql_s_steps = dqn_cql_s_vals['step'].values

            # Plot the IQN contributions for each VaR threshold
            for ivar in range(20):  # Loop over the different alpha settings
                iqn_ns_temp = iqn_ns_vals[iqn_ns_vals.var_index == ivar]
                iqn_s_temp = iqn_s_vals[iqn_s_vals.var_index == ivar]                

                iqn_ns_item = iqn_ns_temp[item].values
                iqn_ns_steps = iqn_ns_temp['step'].values
                iqn_s_item = iqn_s_temp[item].values
                iqn_s_steps = iqn_s_temp['step'].values
                
                sns.lineplot(x=iqn_ns_steps, y=iqn_ns_item, color=ns_colors[ivar], ci=None, linewidth=1, ax=ax)
                sns.lineplot(x=iqn_s_steps, y=iqn_s_item, color=s_colors[ivar], ci=None, linewidth=1, ax=ax)

            # Plot the DQN contributions
            sns.lineplot(x=dqn_ns_steps, y=dqn_ns_item, color='xkcd:bright purple', ci=None, linewidth=1, ax=ax)
            sns.lineplot(x=dqn_s_steps, y=dqn_s_item, color='xkcd:burnt sienna', ci=None, linewidth=1, ax=ax)
            sns.lineplot(x=dqn_cql_ns_steps, y=dqn_cql_ns_item, color='xkcd:royal purple', ci=None, linewidth=1, ax=ax)
            sns.lineplot(x=dqn_cql_s_steps, y=dqn_cql_s_item, color='xkcd:reddish brown', ci=None, linewidth=1, ax=ax)

            ax.set_title(titles[i])
            ax.set_xticks(np.arange(-window_pre, window_post+1))
            ax.axvline(x=0, ls='--', color='gray', alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_ylabel("")

        fig.patch.set_facecolor('white')

        plt.savefig("stats/iqn_value_agg.png")

        # NOW for IQN + CQL
        fig, axs = plt.subplots(2,2, figsize=(12,8), dpi=600, sharex=True)
        
        for i, item in enumerate(['q_dn', 'q_rn', 'v_dn', 'v_rn']):
            ax = axs[i//2, i%2]
            # Extract the DQN results
            dqn_ns_item = dqn_ns_vals[item].values
            dqn_ns_steps = dqn_ns_vals['step'].values
            dqn_s_item = dqn_s_vals[item].values
            dqn_s_steps = dqn_s_vals['step'].values

            # Extract the DQN+CQL results
            dqn_cql_ns_item = dqn_cql_ns_vals[item].values
            dqn_cql_ns_steps = dqn_cql_ns_vals['step'].values
            dqn_cql_s_item = dqn_cql_s_vals[item].values
            dqn_cql_s_steps = dqn_cql_s_vals['step'].values

            # Plot the IQN contributions for each VaR threshold
            for ivar in range(20):  # Loop over the different alpha settings
                iqn_ns_temp = iqn_cql_ns_vals[iqn_cql_ns_vals.var_index == ivar]
                iqn_s_temp = iqn_cql_s_vals[iqn_cql_s_vals.var_index == ivar]                

                iqn_ns_item = iqn_ns_temp[item].values
                iqn_ns_steps = iqn_ns_temp['step'].values
                iqn_s_item = iqn_s_temp[item].values
                iqn_s_steps = iqn_s_temp['step'].values

                sns.lineplot(x=iqn_ns_steps, y=iqn_ns_item, color=ns_colors[ivar], ci=None, linewidth=1, ax=ax)
                sns.lineplot(x=iqn_s_steps, y=iqn_s_item, color=s_colors[ivar], ci=None, linewidth=1, ax=ax)

            # Plot the DQN contributions
            sns.lineplot(x=dqn_ns_steps, y=dqn_ns_item, color='xkcd:bright purple', ci=None, linewidth=1, ax=ax)
            sns.lineplot(x=dqn_s_steps, y=dqn_s_item, color='xkcd:burnt sienna', ci=None, linewidth=1, ax=ax)
            sns.lineplot(x=dqn_cql_ns_steps, y=dqn_cql_ns_item, color='xkcd:royal purple', ci=None, linewidth=1, ax=ax)
            sns.lineplot(x=dqn_cql_s_steps, y=dqn_cql_s_item, color='xkcd:reddish brown', ci=None, linewidth=1, ax=ax)

            ax.set_title(titles[i])
            ax.set_xticks(np.arange(-window_pre, window_post+1))
            ax.axvline(x=0, ls='--', color='gray', alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_ylabel("")

        fig.patch.set_facecolor('white')
        plt.savefig("stats/iqn_cql_value_agg.png")



        



if __name__ == '__main__':
    run()