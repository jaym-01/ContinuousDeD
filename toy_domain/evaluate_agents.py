"""
evaluate_agents.py

SIGNATURE TODO
-------------------------------
Notes:

"""

import torch
from model import IQN, DQN
import numpy as np
import os, random

import MultiPro
import pickle
import gym
import argparse

import LifeGate

# Function for the policy
class fixed_subopt_policy:
    """
    Defines a fixed policy that tries to go through the dead-end region. 

    Uses epsilon parameter for some randomization of the execution
    """
    def __init__(self, epsilon, num_actions=5):
        self.num_actions = num_actions
        self.eps = epsilon

    def get_action(self, state):
        if random.random() <= self.eps:
            return np.random.randint(self.num_actions)
        else:
            if state[0] < 5:
                return 4
            else:
                return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-num_trajs", type=int, default=1000, help="Specify the number of trajectories to sample and test with")
    parser.add_argument("-iqn_agent", type=str, default="lifegate_ded_p2", help="the location of the IQN+CQL agent trained parameters")
    parser.add_argument("-dqn_agent", type=str, default="lifegate_DQN_ded_p2", help="the location of the DQN agent trained parameters")
    parser.add_argument("-num_samples", type=int, default=1000, help="the number of samples drawn from the IQN when evaluating each position")
    parser.add_argument("-eps", type=float, default=0.35, help="The epsilon value that will randomize the predetermined policy")
    parser.add_argument("-info", type=str, default="subopt_pol_eval", help="The experiment name")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs")

    args = parser.parse_args()

    save_dir = "runs/"+args.info
    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    random_state = np.random.RandomState(args.seed)

    distded_pth = args.iqn_agent
    ded_pth = args.dqn_agent

    eps = args.eps
    num_trajectories = args.num_trajs

    # Initialize the models (Q_D, Q_R) for both the IQN and DQN
    state_size = (2,)
    action_size = 5
    layer_size = 512
    n_step = 1
    seed=1
    N = 8
    drm='cvar'
    duel=False
    noisy=False
    device='cpu'

    iqn_d = IQN(state_size, action_size, layer_size, n_step, seed, N, drm=drm, dueling=duel, noisy=noisy, device=device).to(device)
    iqn_r = IQN(state_size, action_size, layer_size, n_step, seed, N, drm=drm, dueling=duel, noisy=noisy, device=device).to(device)

    dqn_d = DQN(state_size, action_size, layer_size, seed, dueling=duel, noisy=noisy, device=device).to(device)
    dqn_r = DQN(state_size, action_size, layer_size, seed, dueling=duel, noisy=noisy, device=device).to(device)

    # Load the parameters of the pretrained models and place them in the initialized models
    iqn_d.load_state_dict(torch.load(os.path.join("runs", distded_pth, distded_pth+'_Qd.pth')))
    iqn_d.eval()
    iqn_r.load_state_dict(torch.load(os.path.join("runs", distded_pth, distded_pth+'_Qr.pth')))
    iqn_r.eval()

    # Set some parameters internal to the IQN model
    iqn_d.eta = 0.001
    iqn_r.eta = 0.001

    dqn_d.load_state_dict(torch.load(os.path.join("runs", ded_pth, ded_pth+'_Qd.pth')))
    dqn_d.eval()
    dqn_r.load_state_dict(torch.load(os.path.join("runs", ded_pth, ded_pth+'_Qr.pth')))
    dqn_r.eval()

    # Initialize the environment
    env = gym.make('LifeGate-v1', state_mode='tabular', rng=random_state, death_drag=0.2, cont_states=True)
    action_size = len(env.legal_actions)
    state_size = (len(env.tabular_state_shape),)

    # Intialize the fixed sub-optimal policy
    policy = fixed_subopt_policy(epsilon=eps)

    # Intialize the number of VaR thresholds we'll use for evaluating the IQN policy
    VaR_thresholds =  np.round(np.linspace(0.05, 1.0, num=20), decimals=2)

    # Loop through each trajectory
    with torch.no_grad():
        output = {}
        for i_traj in range(num_trajectories):
            print(f"Running trajectory {i_traj}")
            step = 0
            done = False
            traj_dict = {'step': [], 'state': [], 'dead_end': [], 'dqn_dn': [], 'dqn_rn': [], 'iqn_dn': [], 'iqn_rn': []}
            state = env.reset()
            while not done: # Not recording the terminal state FYI
                # Intialize cvar arrays
                cvar_dn_traj = np.zeros(len(VaR_thresholds))
                cvar_rn_traj = np.zeros(len(VaR_thresholds))

                # Cast state into torch.tensor to be compatible with Q-functions
                inp_state = torch.from_numpy(np.array(state).reshape((1,-1))).float()
                
                # Gather the values of each state    
                iqn_d_vals, __ = iqn_d.forward(inp_state, args.num_samples, use_drm=False)
                iqn_r_vals, __ = iqn_r.forward(inp_state, args.num_samples, use_drm=False)

                iqn_d_vals = np.sort(iqn_d_vals.squeeze().numpy(), axis=0)  # Sort the values for each action (for CVaR computation)
                iqn_r_vals = np.sort(iqn_r_vals.squeeze().numpy(), axis=0)  # Sort the values for each action (for CVaR computation

                for ii, var in enumerate(VaR_thresholds):
                    var_cutoff = round(var*args.num_samples)
                    cvar_dn_traj[ii] = np.median(np.mean(iqn_d_vals[:var_cutoff, :], axis=0), axis=-1)
                    cvar_rn_traj[ii] = np.median(np.mean(iqn_r_vals[:var_cutoff, :], axis=0), axis=-1)
        
                dqn_d_vals = np.median(dqn_d.forward(inp_state).squeeze(), axis=-1)
                dqn_r_vals = np.median(dqn_r.forward(inp_state).squeeze(), axis=-1)

                # Check to see if state is within dead-end region...
                if (state[0] >= 5) and (state[1] >= 5):
                    de = 1
                else:
                    de = 0

                # Record all of the information!
                traj_dict['step'].append(step)
                traj_dict['state'].append(state)
                traj_dict['dead_end'].append(de)
                traj_dict['iqn_dn'].append(cvar_dn_traj)
                traj_dict['iqn_rn'].append(cvar_rn_traj)
                traj_dict['dqn_dn'].append(dqn_d_vals)
                traj_dict['dqn_rn'].append(dqn_r_vals)

                # Get the action and take a setp in the environment
                action = policy.get_action(state)
                state, _, done, _ = env.step(action)

                step += 1

            # After completing the trajectory add the stats to the output dictionary
            output[i_traj] = traj_dict

        # Save the output dictionary
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "value_data.pkl"), "wb") as f:
            pickle.dump(output, f)







