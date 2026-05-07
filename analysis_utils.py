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

from sklearn.metrics import auc

import torch

#########################################################################################
##     HELPER FUNCTIONS
## Primarily Pandas mapping functions to evaluate the estimated values for the 
##   recorded trajectories with the  D- and R- Networks.
#########################################################################################

class th:
    """Obsolete hard coded thresholds for evaluating the trained D/R-Networks"""
    ded_dn_red = -0.25
    ded_rn_red = 0.75

    ded_dn_yel = -0.15
    ded_rn_yel = 0.85

    ded_dn_gry1 = -0.10
    ded_rn_gry1 = 0.90
    
    ded_dn_gry2 = -0.05
    ded_rn_gry2 = 0.95

    new_dn_red = -0.6
    new_rn_red = 0.4

    new_dn_yel = -0.5
    new_rn_yel = 0.5

def compare_ded_red(row):
    return np.logical_and(row['v_dn']<th.ded_dn_red, row["v_rn"]<th.ded_rn_red)

def compare_ded_yellow(row):
    return np.logical_and(row['v_dn']<th.ded_dn_yel, row["v_rn"]<th.ded_rn_yel)

def compare_red(row):
    return np.logical_and(row['v_dn']<th.new_dn_red, row['v_rn']<th.new_rn_red)
    
def compare_yellow(row):
    return np.logical_and(row['v_dn']<th.new_dn_yel, row['v_rn']<th.new_rn_yel)

def compare_flag_range(row):
    """Loop over the supplied set of possible thresholds and determine when a flag may be raised."""
    # Hard coded range of possible thresholds for learned value functions
    thresh_array = np.linspace(0, 1.0, 1001)  
    output_range = []
    for thr in thresh_array:
        dn_thr = thr-1.0  # Threshold for the D-network is inverted from the R-Network
        output_range.append(np.logical_and(row["v_dn"]<dn_thr, row["v_rn"]<thr))
    return output_range if isinstance(row['v_dn'],float) else np.vstack(output_range)


def extract_first_relevant_ts(group):
    # Reset the index of the group (if we can...)
    tmp_grp = group.reset_index(drop=True)
    # Check whether this is an AHE patient (empty sepsis time)
    if tmp_grp['m:time_from_sepsis_h'].isna().all():
        pt_type = "AHE"
        col_of_interest = 'o:mbp'
        threshold = 65
        comparison = tmp_grp[tmp_grp[col_of_interest] <= threshold]
    else:  # Patient was septic, we're looking for the index of presumed onset (may occur early)
        pt_type = "Sepsis"
        col_of_interest = 'm:bloc'  # bloc==0 is the presumed time of onset        
        # col_of_interest = 'm:time_from_sepsis_h'
        threshold = 0
        comparison = tmp_grp[tmp_grp[col_of_interest] == threshold]
        # We just need the positive number of hours from beginning of trajectory to presumed onset
        # time_of_sepsis_onset = abs(round(tmp_grp.iloc[0][col_of_interest])) 
        
    if len(comparison)>0:
        ts = comparison.index[0] - len(group)
    else:
        ts = None
    # ts = time_of_sepsis_onset - len(group)
    
    return (ts, pt_type)

################################### 
#    EVALUATE TRAINED MODELS
###################################

def get_dn_rn_info(qnet_dn, qnet_rn, encoded_data, device, distributional=False, num_samples=1000):
    """
    Evaluate the trained networks with to produce the Q-values for the encoded test data.

    Args:
        q_net_dn (torch.nn.Module): The trained D-Network
        q_net_rn (torch.nn.Module): The trained R-Network
        encoded_data (numpy npz file): The encoded test data with fields ['states', 'actions', 'rewards', 'lengths']
        device (torch.device): CPU or CUDA
        distributional (boolean): Whether the Q-functions are distributional or not

    Returns:
        blech
    """
    traj_indices = range(len(encoded_data['states']))
    data = {'traj': [], 'step': [], 's': [], 'a': [], 'q_dn': [], 'q_rn': [], 'category': [], 'stay_id': []}
    print("Making Q-values...")
    for traj in traj_indices:
        traj_len = int(encoded_data['lengths'][traj][0])
        traj_sid = encoded_data['stay_ids'][traj]
        traj_states = torch.from_numpy(encoded_data['states'][traj][:traj_len]).to(device)
        if not distributional:
            traj_q_dn = qnet_dn.get_q(traj_states)
            traj_q_rn = qnet_rn.get_q(traj_states)
        else:
            # Outputs from the IQN network are BATCHxNUMSAMPLESxNUMACTIONS, we sort the samples for better CVAR estimation
            traj_q_dn = np.sort(qnet_dn.estimate_q_dist(traj_states, num_samples).detach().cpu().numpy(), axis=1)
            traj_q_rn = np.sort(qnet_rn.estimate_q_dist(traj_states, num_samples).detach().cpu().numpy(), axis=1)
        traj_q_dn = np.clip(traj_q_dn, -1, 0)
        traj_q_rn = np.clip(traj_q_rn, 0, 1)
        traj_r = encoded_data['rewards'][traj][:traj_len]
        traj_a = encoded_data['actions'][traj][:traj_len]
        
        for i, action in enumerate(traj_a):
            data['traj'].append(traj)
            data['step'].append(i)
            data['s'].append(traj_states[i, :].detach().cpu().numpy())
            data['a'].append(int(action))
            data['stay_id'].append(traj_sid)
            if not distributional:
                data['q_dn'].append(traj_q_dn[i, :])
                data['q_rn'].append(traj_q_rn[i, :])
            else:
                data['q_dn'].append(traj_q_dn[i,...])
                data['q_rn'].append(traj_q_rn[i,...])
            if traj_r[-1] == -1.0:
                data['category'].append(-1)
            elif traj_r[-1] == 1.0:
                data['category'].append(1)
            else:
                raise ValueError('last reward of a trajectory is neither of -+1.')
    print("Q values made.")
    return data

#Fixed version for use with continuous IQN
def pre_flag_splitting(value_data, VaR_thresholds, distributional=False):
    if distributional:
        num_VaR_thres = len(VaR_thresholds)
    else:
        num_VaR_thres = 0

    value_data = pd.DataFrame(value_data)

    results = {"survivors": {}, "nonsurvivors": {}}
    non_survivor_trajectories = sorted(value_data[value_data.category == -1].traj.unique().tolist())
    survivor_trajectories = sorted(value_data[value_data.category == 1].traj.unique().tolist())
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
            d = value_data[value_data.traj == traj]
            sid_traj.append(d.stay_id.tolist()[0])  # We only need one Stay ID per trajectory (was repeated across time for each trajectory in this column)
            dn_q_traj.append(np.array(d.q_dn.tolist(), dtype=np.float32))
            rn_q_traj.append(np.array(d.q_rn.tolist(), dtype=np.float32))
            if not distributional:
                dn_q_selected_action = [d.q_dn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_dn.shape[0])] 
                rn_q_selected_action = [d.q_rn.tolist()[t][d.a.tolist()[t]] for t in range(d.q_rn.shape[0])] 
            else:
                (num_steps, num_samples, num_actions) = dn_q_traj[-1].shape
                cvar_dn_traj = np.zeros((num_steps, num_VaR_thres, num_actions))
                cvar_rn_traj = np.zeros((num_steps, num_VaR_thres, num_actions))
                # Loop through all VaR thresholds and the extract the selected action
                for i, var in enumerate(VaR_thresholds):
                    var_cutoff = round(var*num_samples)
                    cvar_dn_traj[:, i, :] = np.mean(dn_q_traj[-1][:, :var_cutoff, :], axis=1)
                    cvar_rn_traj[:, i, :] = np.mean(rn_q_traj[-1][:, :var_cutoff, :], axis=1)

                # In continuous spaces, the network already evaluated the specific action.
                # cvar_traj has shape (num_steps, num_VaR_thres, 1). We just grab index 0 of the last dimension.
                dn_q_selected_action = cvar_dn_traj[..., 0]
                rn_q_selected_action = cvar_rn_traj[..., 0]

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
    
    return results

def create_analysis_df(results, num_survivors, num_nonsurvivors):
    """
    Create an indexable dataframe from the results data processed in `eval_rl.py`.
    
    Args:
        results (dict): a nested dictionary keyed by [nonsurvivors, survivors]x[traj] that contains the computed value of sequential state x treatment pairs
        num_survivors (int): the number of surviving patient trajectories
        num_nonsurvivors (int): the number of non_surviving patient trajectories

    Returns:
        surv_df (pd.DataFrame): A dataframe indexing the trajectory and step number for surviving patient trajectories
        nonsurv_df (pd.DataFrame): A dataframe indexing the trajectory and step number for nonsurviving patient trajectories
    """
    # Initialize the dictionaries that will be transformed into dataframes 
    surv_data = {'traj': [], 'step': [], 'q_dn': [], 'q_rn': [], 'v_dn': [], 'v_rn': []}
    nonsurv_data = {'traj': [], 'step': [], 'q_dn': [], 'q_rn': [], 'v_dn': [], 'v_rn': []}

    # Populate the dictionaries
    # First, the surviving patients
    surv_results = results['survivors']
    for traj in range(num_survivors):
        traj_qd = surv_results['dn_q_selected_action_traj'][traj]
        traj_qr = surv_results['rn_q_selected_action_traj'][traj]
        traj_vd = surv_results['dn_v_median_traj'][traj]
        traj_vr = surv_results['rn_v_median_traj'][traj]

        for i in range(len(traj_qd)):
            surv_data['traj'].append(traj)
            surv_data['step'].append(i-len(traj_qd))
            surv_data['q_dn'].append(traj_qd[i])
            surv_data['q_rn'].append(traj_qr[i])
            surv_data['v_dn'].append(traj_vd[i])
            surv_data['v_rn'].append(traj_vr[i])

    # Next, for the nonsurviving patients
    nonsurv_results = results['nonsurvivors']
    for traj in range(num_nonsurvivors):
        traj_qd = nonsurv_results['dn_q_selected_action_traj'][traj]
        traj_qr = nonsurv_results['rn_q_selected_action_traj'][traj]
        traj_vd = nonsurv_results['dn_v_median_traj'][traj]
        traj_vr = nonsurv_results['rn_v_median_traj'][traj]

        for i in range(len(traj_qd)):
            nonsurv_data['traj'].append(traj)
            nonsurv_data['step'].append(i-len(traj_qd))
            nonsurv_data['q_dn'].append(traj_qd[i])
            nonsurv_data['q_rn'].append(traj_qr[i])
            nonsurv_data['v_dn'].append(traj_vd[i])
            nonsurv_data['v_rn'].append(traj_vr[i])

    # Create the DataFrames from the populated dictionaries
    surv_df = pd.DataFrame(surv_data)
    nonsurv_df = pd.DataFrame(nonsurv_data)

    return surv_df, nonsurv_df


def compute_auc(surv_df, nonsurv_df, num_survivors, num_nonsurvivors, iqn_size=None):
    """
    Analyze whether or not each trajectory raised a flag in a true positive or false positive nature,
      compute the TPR and FPR, compute the AUC.

    True Positive Rate: Proportion of non-surviving trajectories that raised a flag
    False Positive Rate: Proportion of surviving trajectories that raised a flag

    Args:
        surv_df: A dataframe constructed of the per-trajectory, per-step value estimates for both D- and R-Networks
            derived from surviving patients
        nonsurv_df: A dataframe constructed of the per-trajectory, per-step value estimates for both D- and R-Networks
            derived from non-surviving patients
        num_survivors: The integer number of surviving patients
        num_nonsurvivors: The integer number of non-surviving patients
        iqn_size: If not None, this is the number of distinct CVaR thresholds used to analyze an IQN trained D-/R-Network
    
    Returns:
        fpr: A numpy array of the false positive rate over a range of possible thresholds used to determine flags
        tpr: A numpy array of the true positive rate over a range of possible thresholds used to determine flags
        auc_out: The calculated AUC for the computed TPR, FPR arrays

    """

    # Assess the flags with a range of thresholds to compute the TPR and FPR
    # TPR: %-age of NS trajectories that are flagged
    # FPR: %-age of Surv trajectories that are flagged...

    if iqn_size is not None:
        surv_data = np.zeros((num_survivors, 1001, iqn_size))
        nonsurv_data = np.zeros((num_nonsurvivors, 1001, iqn_size))
    else:
        surv_data = np.zeros((num_survivors, 1001))
        nonsurv_data = np.zeros((num_nonsurvivors, 1001))

    # Loop over the non-survivor trajectories
    for traj in range(num_nonsurvivors): #range(4): #
        ns_dt = nonsurv_df[nonsurv_df.traj==traj]

        comp_flags = np.stack(ns_dt.apply(compare_flag_range, axis=1).values)
        
        # Pull off whether there was a flag raised for this trajectory, for all thresholds
        if iqn_size is not None:
            # For each alpha value of the CVaR computation, evaluate each trajectory
            for ii in range(iqn_size):
                nonsurv_data[traj, :, ii] = np.any(comp_flags[...,ii], axis=0)
        else:
            nonsurv_data[traj,:] = np.any(comp_flags, axis=0)

    # Loop over the survivor trajectories
    for traj in range(num_survivors):
        s_dt = surv_df[surv_df.traj==traj]

        comp_flags = np.stack(s_dt.apply(compare_flag_range, axis=1).values)
        
        # Pull off whether there was a flag raised for this trajectory, for all thresholds
        if iqn_size is not None:
            # For each alpha value of the CVaR computation, evaluate each trajectory for all thresholds
            for ii in range(iqn_size):
                surv_data[traj, :, ii] = np.any(comp_flags[...,ii], axis=0)
        else:
            surv_data[traj,:] = np.any(comp_flags, axis=0)

    # With the trajectories evaluated, compute the FPR and TPR
    fpr = np.sum(surv_data, axis=0) / num_survivors
    tpr = np.sum(nonsurv_data, axis=0) / num_nonsurvivors

    # Sort the FPR and TPR arrays, compute the AUC
    if iqn_size is not None:
        auc_out = np.zeros(iqn_size)
        for ii in range(iqn_size):
            fpr_ii_idx = np.argsort(fpr[:, ii])
            fpr[:, ii] = fpr[fpr_ii_idx, ii]
            tpr[:, ii] = tpr[fpr_ii_idx, ii]
            auc_out[ii] = auc(fpr[:, ii], tpr[:, ii])
    else:
        fpr_idx = np.argsort(fpr)
        fpr = fpr[fpr_idx]
        tpr = tpr[fpr_idx]
        auc_out = auc(fpr, tpr)

    return fpr, tpr, [auc_out]


def compute_diff_in_flags(iqn_nonsurv_df, iqn_surv_df, dqn_nonsurv_df, dqn_surv_df, num_survivors, num_nonsurvivors, iqn_size=20):
    # Initilize dictionaries for flag computation...  # Not worrying about onset for now...
    surv_data = {'traj': [], 'pt_type': [], 'dqn_flag': [], 'iqn_flags': []} # , 'presumed_onset': []}
    nonsurv_data = {'traj': [], 'pt_type': [], 'dqn_flag': [], 'iqn_flags': []} # , 'presumed_onset': []}

    # Compute the baseline/static flags (single threshold) for each non-surviving trajectory (computed for both DQN and IQN based D/R-Network)
    for traj in range(num_nonsurvivors):
        dqn_ns_dt = dqn_nonsurv_df[dqn_nonsurv_df.traj==traj]
        iqn_ns_dt = iqn_nonsurv_df[iqn_nonsurv_df.traj==traj]

        dqn_ns_flags = np.vstack(dqn_ns_dt.apply(compare_ded_yellow, axis=1).values)
        iqn_ns_flags = np.vstack(iqn_ns_dt.apply(compare_yellow, axis=1).values)

        pt_type = iqn_ns_dt.iloc[0]['pt_type']

        try:
            dqn_fst_flg = np.where(dqn_ns_flags==True)[0][0]+dqn_ns_df.iloc[0]['step']
        except:
            dqn_fst_flg = None

        first_flags_iqn = []
        for ii in range(iqn_size):
            try:
                fst_flg = np.where(iqn_ns_flags[:, ii] == True)[0][0]+iqn_ns_dt.iloc[0]['step']
            except:
                fst_flg = None
            first_flags_iqn.append(fst_flg)

        nonsurv_data['traj'].append(traj)
        nonsurv_data['pt_type'].append(pt_type)
        nonsurv_data['dqn_flag'].append(dqn_fst_flg)
        nonsurv_data['iqn_flags'].append(first_flags_iqn)

    for traj in range(num_survivors):
        dqn_s_dt = dqn_surv_df[dqn_surv_df.traj==traj]
        iqn_s_dt = iqn_surv_df[iqn_surv_df.traj==traj]

        dqn_s_flags = np.vstack(dqn_s_dt.apply(compare_ded_yellow, axis=1).values)
        iqn_s_flags = np.vstack(iqn_s_dt.apply(compare_yellow, axis=1).values)

        pt_type = iqn_s_dt.iloc[0]['pt_type']

        try:
            dqn_fst_flg = np.where(dqn_s_flags==True)[0][0]+dqn_s_dt.iloc[0]['step']
        except:
            dqn_fst_flg = None

        first_flags_iqn = []
        for ii in range(iqn_size):
            try:
                fst_flg = np.where(iqn_s_flags[:, ii]==True)[0][0]+iqn_s_dt.iloc[0]['step']
            except:
                fst_flg = None
            first_flags_iqn.append(fst_flg)

        surv_data['traj'].append(traj)
        surv_data['pt_type'].append(pt_type)
        surv_data['dqn_flag'].append(dqn_fst_flg)
        surv_data['iqn_flags'].append(first_flags_iqn)


    # Create the DataFrames from the populated dictionaries for the first flags
    flags_s_df = pd.DataFrame(surv_data)
    flags_ns_df = pd.DataFrame(nonsurv_data)

    diff_in_flags_s = []
    diff_in_flags_ns = []

    num_AHE_s = flags_s_df[flags_s_df.pt_type == 'AHE'].shape[0]
    num_AHE_ns = flags_ns_df[flags_ns_df.pt_type == 'AHE'].shape[0]

    for pt_typ in ['AHE', 'Sepsis']:
        typ_s_df = flags_s_df[flags_s_df.pt_type == pt_typ]
        typ_ns_df = flags_ns_df[flags_ns_df.pt_type == pt_typ]

        for traj in typ_s_df.traj.values:
            traj_flag_diffs = []
            dqn_flag_ts = typ_df.loc[traj]['dqn_flag']
            if np.isnan(dqn_flag_ts):
                dqn_flag_ts = 0
            for ivar in range(iqn_size):
                ivar_flag_ts = typ_s_df.loc[traj]['iqn_flags'][ivar]
                if ivar_flag_ts is None:
                    traj_flag_diffs.append(np.nan)
                else:
                    traj_flag_diffs.append(dqn_flag_ts - ivar_flag_ts)

            diff_in_flags_s.append(traj_flag_diffs)

        for traj in typ_ns_df.traj.values:
            traj_flag_diffs = []
            dqn_flag_ts = typ_df.loc[traj]['dqn_flag']
            if np.isnan(dqn_flag_ts):
                dqn_flag_ts = 0
            for ivar in range(iqn_size):
                ivar_flag_ts = typ_ns_df.loc[traj]['iqn_flags'][ivar]
                if ivar_flag_ts is None:
                    traj_flag_diffs.append(np.nan)
                else:
                    traj_flag_diffs.append(dqn_flag_ts - ivar_flag_ts)

            diff_in_flags_ns.append(traj_flag_diffs)

    return diff_in_flags_s, diff_in_flags_ns, num_AHE_s, num_AHE_ns

