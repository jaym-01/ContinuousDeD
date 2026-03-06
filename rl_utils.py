import operator
import gc

import numpy as np
import pandas as pd


class _DRM:
    """Picklable distortion risk measure callable."""
    def __init__(self, name, eta):
        self.name = name.lower()
        self.eta = eta

    def __call__(self, tau):
        eta = self.eta
        if self.name == 'cvar':
            return tau * eta
        elif self.name == 'cpw':
            return tau**eta / (tau**eta + (1 - tau)**eta)**(1 / eta)
        elif self.name == 'identity':
            return tau
        elif self.name == 'power':
            if eta <= 0:
                return 1 - (1 - tau)**(1 / (1 + abs(eta)))
            else:
                return tau**(1 / (1 + abs(eta)))
        else:
            raise Exception("The supplied DRM is not implemented. Current options: {CVaR, CPW, Identity, Power}")

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ncde_utils import create_net
from analysis_utils import get_dn_rn_info, pre_flag_splitting, create_analysis_df, compute_auc


##########################################################
#              HELPER FUNCTIONS
##########################################################

def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wise depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    #assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss

class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    
    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)

    
##########################################################
#     LOAD ENCODED DATA AND PUT IN S,A,S',R FORMAT
##########################################################

class RLDataLoader(object):
    def __init__(self, data_dir, rng, minibatch_size, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0, drop_smaller_than_minibatch=False, dataset='train', device='cpu'):
        '''
        Custom DataLoader that ensures that transitons from both pos/neg trajectories are sampled in a balanced way.

        Args:
            data_dir (str): pointer to the location of the .npz files where the encoded data is stored
            rng (random): the random number generator that was established for reproducibility
            minibatch_size (int): the size of minibatches that will be sampled using this dataloader
            drop_smaller_than_minibatch (int?): TBD?
            dataset (str): default='train', options = ['train', 'validation', 'test', 'overlap']

        Use: first call make_transition_train_data() once, then call reset() before each epoch of getting all minibatches
        '''
        self.rng = rng
        self.minibatch_size = minibatch_size
        self.drop_smaller_than_minibatch = drop_smaller_than_minibatch
        self.ps = pos_samples_in_minibatch  # The number of positive terminal transtions placed in each minibatch
        self.ns = neg_samples_in_minibatch  # The number of negative terminal transitons place in each minibatch
        self.device = device
        # Load the data (encoded data is stored as .npz with files ['states', 'actions', 'rewards', 'lengths'])
        if dataset == 'train_val': # Combine the training and validation dataset (final training of best performing model...)
            tr_data = np.load(data_dir+"encoded_train.npz")
            v_data = np.load(data_dir+"encoded_validation.npz")

            # Combine the encoded states
            self.encoded_data = np.vstack((tr_data['states'], v_data['states']))
            # Combine the actions, ensure that they're integers and remove unncessary dims
            self.encoded_actions = np.vstack(
                (
                    tr_data['actions'].astype(int).squeeze(),
                    v_data['actions'].astype(int).squeeze()
                )
            )
            # Combine the rewards, remove unecessary dims
            self.encoded_rewards = np.vstack((tr_data['rewards'].squeeze(), v_data['rewards'].squeeze()))
            # Combine the recorded lengths of each trajectory, change to integers (for indexing purposes)
            self.encoded_lengths = np.vstack((tr_data['lengths'].astype(int), v_data['lengths'].astype(int))).squeeze()
        else:
            encoded_data_file = np.load(data_dir+"encoded_{}.npz".format(dataset))
            self.encoded_data = encoded_data_file['states']
            # Ensure that the actions are recorded as integers since they'll be used to index the Q-value returns
            self.encoded_actions = encoded_data_file['actions'].astype(int).squeeze()  # Removing unneeded extra final dimension... If (N, T, 1) -> (N, T)... If (N,T,num_actions) (one-hot encoded), then nothing changes
            self.encoded_rewards = encoded_data_file['rewards'].squeeze()  # Removing unneeded extra final dimension... (N, T, 1) -> (N,T)
            # Change the recorded lengths of each trajectory since this quantity is used to index the other arrays
            self.encoded_lengths = encoded_data_file['lengths'].astype(int).squeeze() # Removing unneeded extra final dimension... (N, 1) -> (N,)

        # Extract the dimension of the encoded states to return for building the Q-networks
        self.data_dim = self.encoded_data.shape[-1]

        # Set up data structures and indexing mechanisms that will be used to sample minibatches
        self.transition_data = {}
        self.transition_indices_pos_last = []
        self.transition_indices_neg_last = []
        self.transition_indices = None
        self.transition_data_size = None
        self.transitions_head = None
        self.transitions_head_pos = None
        self.transitions_head_neg = None
        self.epoch_finished = True  # to enforce reset() before use
        self.epoch_pos_finished = True
        self.epoch_neg_finished = True
        self.num_minibatches_epoch = None
    
    def reset(self, shuffle):
        if shuffle:
            self.rng.shuffle(self.transition_indices)
            self.rng.shuffle(self.transition_indices_pos_last)
            self.rng.shuffle(self.transition_indices_neg_last)
        # Reset the index/counters for traversing the stored data arrays when "sampling" minibatches
        self.transitions_head = 0
        self.transitions_head_pos = 0
        self.transitions_head_neg = 0
        # Reset the flag indicators for when we've exhausted the available data
        self.epoch_finished = False
        self.epoch_pos_finished = False
        self.epoch_neg_finished = False

    def make_transition_data(self, release=False):
        print("DataLoader: making transitions (s,a,r,s') from encoded data structures")
        self.transition_data['s'] = {}
        self.transition_data['actions'] = {}
        self.transition_data['rewards'] = {}
        self.transition_data['next_s'] = {}
        self.transition_data['terminals'] = {}
        indices_pos = []
        indices_neg = []
        counter = 0
        
        for traj in range(self.encoded_data.shape[0]):
            # Check whether terminal reward for this trajectory is positive (patient survived)
            traj_is_positive = self.encoded_rewards[traj, self.encoded_lengths[traj]-1] > 0
            for t in range(self.encoded_lengths[traj] - 1):
                self.transition_data['s'][counter] = self.encoded_data[traj,t, :]
                self.transition_data['next_s'][counter] = self.encoded_data[traj, t+1, :]
                self.transition_data['actions'][counter] = self.encoded_actions[traj, t]
                self.transition_data['rewards'][counter] = self.encoded_rewards[traj, t]
                self.transition_data['terminals'][counter] = 0
                if traj_is_positive: 
                    indices_pos.append(counter)
                else:
                    indices_neg.append(counter)
                counter += 1
            # For the last transition in the trajectory
            tlast = self.encoded_lengths[traj] - 1 # Get the index of the terminal state
            self.transition_data['s'][counter] = self.encoded_data[traj, tlast, :]
            self.transition_data['next_s'][counter] = np.zeros_like(self.encoded_data[traj, tlast, :])
            self.transition_data['actions'][counter] = self.encoded_actions[traj, tlast]
            self.transition_data['rewards'][counter] = self.encoded_rewards[traj, tlast]
            self.transition_data['terminals'][counter] = 1
            if traj_is_positive:
                self.transition_indices_pos_last.append(counter)
            else:
                self.transition_indices_neg_last.append(counter)
            counter += 1
        self.transition_data_size = counter
        self.transition_indices = np.arange(self.transition_data_size)
        if release:
            del self.encoded_data, self.encoded_actions, self.encoded_rewards
            self.encoded_data, self.encoded_actions, self.encoded_rewards = None, None, None
            gc.collect()

        # Compute the number of minibatches that will be drawn per epoch
        self.num_minibatches_epoch = int(np.floor(self.transition_data_size / self.minibatch_size)) + int(1 - self.drop_smaller_than_minibatch)
    
    def get_next_minibatch(self):
        if self.epoch_finished == True:
            print('Epoch finished, please call reset() method before next call to get_next_minibatch()')
            return None
        # Getting data from dictionaries
        offset = self.ns + self.ps  # The number of samples we'll need to replace in the list of indices below
        minibatch_main_index_list = list(self.transition_indices[self.transitions_head:self.transitions_head + self.minibatch_size - offset])
        # Throughout an epoch, we cycle through the terminal transitions and include them specially in the sampled minibatch as desired
        # self.ps is an integer number of positive terminal states added to the minibatch
        # self.ns is an integer number of negative terminal states added to the minibatch
        minibatch_pos_last_index_list = self.transition_indices_pos_last[self.transitions_head_pos:self.transitions_head_pos + self.ps]
        minibatch_neg_last_index_list = self.transition_indices_neg_last[self.transitions_head_neg:self.transitions_head_neg + self.ns]
        # Increment the starting index for the next terminal state resampling
        self.transitions_head_pos += self.ps
        self.transitions_head_neg += self.ns
        # Combine the index lists to construct our minibatch from using operator.itemgetter
        minibatch_index_list = minibatch_main_index_list + minibatch_pos_last_index_list + minibatch_neg_last_index_list
        get_from_dict = operator.itemgetter(*minibatch_index_list)
        
        # Extract the transition data to form the minibatch
        # operator.itemgetter just appends the extracted subarrays in a tuple, wrap in np.float32 to get ndarrays
        # We also cast these elements to torch Tensors and move to the computation device so they're ready for processing right away
        s_minibatch = torch.from_numpy(np.float32(get_from_dict(self.transition_data['s']))).to(self.device)
        actions_minibatch = torch.from_numpy(np.float32(get_from_dict(self.transition_data['actions']))).to(self.device, dtype=torch.int64)
        rewards_minibatch = torch.from_numpy(np.float32(get_from_dict(self.transition_data['rewards']))).to(self.device)
        next_s_minibatch = torch.from_numpy(np.float32(get_from_dict(self.transition_data['next_s']))).to(self.device)
        terminals_minibatch = torch.from_numpy(np.float32(get_from_dict(self.transition_data['terminals']))).to(self.device)
        
        # Updating current data head to start the next minibatch at the end of the current one
        self.transitions_head += self.minibatch_size
        self.epoch_finished = self.transitions_head + self.drop_smaller_than_minibatch*self.minibatch_size >= self.transition_data_size
        self.transitions_head_pos = self.transitions_head_pos % len(self.transition_indices_pos_last)
        self.transitions_head_neg = self.transitions_head_neg % len(self.transition_indices_neg_last)

        return s_minibatch, actions_minibatch, rewards_minibatch, next_s_minibatch, terminals_minibatch, self.epoch_finished




##########################################################
#           AGENT DEFINITIONS (DDQN and IQN)
##########################################################

class DQN_Agent(nn.Module): 
    """Simple Deep Q-Network using Double Q-Learning via use of a Target Network.
    
    Args:
        TBD

    Returns:
        TBD
    """

    def __init__(self, state_size, params, sided_Q='negative', device="cuda:0"):
        super(DQN_Agent, self).__init__()

        # Establish the parameters of learning the value function with this Network
        self.seed = torch.manual_seed(params.get("seed", 0))
        self.state_dim = state_size
        self.action_size = params.get("num_actions", 25)
        self.layer_size = params.get("num_q_hidden_units", 64)
        self.num_layers = params.get("num_q_layers", 1)
        self.gamma = params.get("gamma", 1.0)
        self.lr = params.get("lr", 5e-3)
        self.use_ddqn = params.get("use_ddqn", True)
        self.tau = params.get("tau", 0.005)
        self.sided_q = sided_Q
        self.device = device

        self.use_cql = params.get("use_cql", False)
        self.min_q_weight = params.get("cql_weight", 0.0)

        # Set up the Q-networks (both primal and target)
        self.network = create_net(state_size, self.action_size, n_layers=self.num_layers, n_units=self.layer_size, nonlinear=nn.ReLU)
        self.target_network = create_net(state_size, self.action_size, n_layers=self.num_layers, n_units=self.layer_size, nonlinear=nn.ReLU)

        self.network.to(self.device)
        self.target_network.to(self.device)
        self.update_counter = 0
        self.update_freq = params.get("q_update_freq", 25)

        # Set up the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, amsgrad=True)

    def _train_on_batch(self, s, a, r, s2, t):
        """With the provided batch of data, run a training Bellman update."""
        
        # Pass input states to the network
        q = self.network(s)
        q2 = self.target_network(s2).detach()  # Begin target construction
        
        # Get the values of the actions used with the corresponding states
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 

        if self.use_ddqn: # Apply the DDQN update
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        
        # Construct the DeD target according to how we want mask the specified values
        if self.sided_q == 'negative':
            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1-t)
        elif self.sided_q == 'positive':
            bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1-t)
        else:
            bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1-t)

        # Calculate the loss (using Huber loss)
        td_error = (bellman_target - q_pred).unsqueeze(1)
        huber_l = calculate_huber_loss(td_error, 1.0)

        loss = huber_l.mean()

        if self.use_cql:
            # TODO -- DEBUG THIS AND SEE WHAT'S GOING ON -- Seems that the working definition requires a change from the author's implementation (using only DistRL)
            """ Code snippet from BY571/CQL/DQN/agent.py
            states, actions, rewards, next_states, dones = experiences
            with torch.no_grad():
                Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
                Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            Q_a_s = self.network(states)
            Q_expected = Q_a_s.gather(1, actions)
            
            cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()
            """
            # Compute the logsumexp of the Q-value "distribution" over actions
            min_q_loss = torch.logsumexp(q.squeeze(), -1).mean() # Average over the batch dimension

            # Compute the CQL penalty by subtracting `min_q_loss` by the average expected Q-value of the observed actions
            # Weighted by the alpha parameter, static and fixed since we're doing Q-learning
            # penalty = (min_q_loss - q_pred.squeeze().mean()) * self.min_q_weight  # QR-DQN/IQN based method... 
            penalty = (min_q_loss - q.squeeze().mean()) * self.min_q_weight  # Inspired by BY571's implementation and demonstration that this works for DQN... *shrug*

            # Adjust the loss according to the CQL penalty
            loss = loss + penalty

        # Apply the gradient update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1) # Clip the gradients...
        self.optimizer.step()

        return loss.detach().cpu().numpy()


    def get_loss(self, s, a, r, s2, t):
        """Calculate the loss without gradients. (For diagnostic purposes / validation)"""
        
        with torch.no_grad():
            # Pass input states to the network
            q = self.network(s).detach()
            q2 = self.target_network(s2).detach()  # Begin target construction
        # Get the values of the actions used with the corresponding states
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 

        if self.use_ddqn: # Apply the DDQN update
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        
        # Construct the DeD target according to how we want mask the specified values
        if self.sided_q == 'negative':
            bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1-t)
        elif self.sided_q == 'positive':
            bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1-t)
        else:
            bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1-t)

        # Calculate the loss (using Huber loss)
        td_error = (bellman_target - q_pred).unsqueeze(1)
        huber_l = calculate_huber_loss(td_error, 1.0)

        loss = huber_l.mean()

        if self.use_cql:
            
            # Compute the logsumexp of the Q-value "distribution" over actions
            min_q_loss = torch.logsumexp(q.squeeze(), -1).mean() # Average over the batch dimension

            # Compute the CQL penalty by subtracting `min_zf_loss` by the average expected Q-value of the observed actions
            # Weighted by the alpha parameter, static and fixed since we're doing Q-learning (as opposed to double gradient descent in Actor-critic formulations...)
            penalty = (min_q_loss - q_pred.squeeze().mean()) * self.min_q_weight

            # Adjust the loss according to the CQL/CODAC penalty
            loss = loss + penalty

        return loss.detach().cpu().numpy()

    def get_q(self, s):
        """ Get the Q-values with the provided states `s`."""
        s = s.to(self.device)

        return self.network(s).detach().cpu().numpy()

    def _get_max_action(self, s):
        """ Using the current Q-Network, get the max action for the provided states `s`."""
        s = s.to(self.device)
        q = self.network(s).detach()
        return q.max(1)[1].cpu().numpy()

    def get_action(self, states):
        return self._get_max_action(states)

    def learn(self, s, a, r, s2, term):
        """Learning from one minibatch."""
        loss = self._train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.soft_update(self.network, self.target_network)
            # self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return loss

    def dump_netork(self, weights_file_path):
        try:
            torch.save(self.network.state_dict(), weights_file_path)
        except:
            pass

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    def resume(self, network_state_dict, target_network_state_dict, optimizer_state_dict):
        self.network.load_state_dict(network_state_dict)
        self.target_network.load_state_dict(target_network_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters (Polyak averaging).
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    @staticmethod
    def weight_transfer(self, from_model, to_model):
        """Directly copies the local network parameters to the target network."""
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # Not pickle-able
        return _dict


class IQN(nn.Module):
    """Base Implicit Quantile Network (from https://arxiv.org/abs/1806.06923) used inside IQN_Agent"""
    def __init__(self, state_size, action_size, layer_size, n_step, N=32, K=32, drm="identity", eta=0.71, use_cql=False, dueling=False, noisy=False, device="cuda:0"):
        """TODO Signature...
        """
        super(IQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N
        self.K = K
        self.n_cos = 64  # Hard coded here...
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device)
        self.dueling = dueling
        self.device = device
        self.drm = self._get_drm(drm, eta)
        self.eta = eta

        # CQL penalty?
        self.use_cql = use_cql

        if noisy:
            layer = NoisyLayer
        else:
            layer = nn.Linear

        # Network architecture
        self.head = layer(self.input_shape, layer_size)
        self.cos_embedding = layer(self.n_cos, layer_size)
        self.ff_1 = layer(layer_size, layer_size)
        self.cos_layer_out = layer_size
        if dueling:
            self.advantage = layer(layer_size, action_size)
            self.value = layer(layer_size, 1)
        else:
            self.ff_2 = layer(layer_size, action_size)

    
    def _get_drm(self, drm, eta=0.71):
        """Define the Distortion Risk Measure Function to resample the taus.

        Args:
            drm (str): The string representations of the DRM we want to use.
            eta (float): The scaling factor for the DRM> Default (0.71) is for CPW.
        """
        return _DRM(drm, eta)

    def calc_cos(self, batch_size, n_tau=32, use_drm=False):
        """
        Calculating the cosin values depending on the number of tau samples
        """
        # Initialize the tau samples 
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1)  # (batch_size, n_tau, 1)
        if use_drm: # Distort the tau samples if desired
            taus = torch.from_numpy(self.drm(taus.numpy())).type(torch.float)
        # Project the tau samples with the cosine transform
        cos = torch.cos(taus.to(self.device)*self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos)
        return cos, taus

    def forward(self, input, num_tau=32, use_drm=False):
        """
        Quantile calculation depending on the number of tau samples.

        Return:
            quantiles: shape of (batch_size, num_tau, action_size)
            taus : shape of (batch_size, num_tau, 1)
        """
        batch_size = input.shape[0]
        x = torch.relu(self.head(input))
        # Sampled the taus and compute the cosine embedding 
        cos, taus = self.calc_cos(batch_size, num_tau, use_drm)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out)

        # x has shape (batch_size, layer_size) for multiplication -> reshape to (batch_size, 1, layer_size)
        x = (x.unsqueeze(1)*cos_x).view(batch_size, num_tau, self.cos_layer_out)
        x = torch.relu(self.ff_1(x))

        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value+advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.ff_2(x)

        return out.view(batch_size, num_tau, self.action_size), taus

    def get_qvalues(self, inputs, act=False, use_drm=False):
        if act:
            quantiles, _ = self.forward(inputs, self.K, use_drm)
        else:
            quantiles, _ = self.forward(inputs, self.N, use_drm)
        
        return quantiles.mean(dim=1)


class IQN_Agent(nn.Module):
    """Initialize an Agent object.
    
    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        network (str): description of the components put together for the IQN model training.
        layer_size (int): size of the hidden layer
        n_step (int): The number of steps ahead for Q-value estimation
        risk_measure (str): The distortion risk measure that we want to use to constrain sampling from the Value distribution
        sided_Q (str): Either 'positive', 'negative' or 'both'. This determines how we'll constrain the Q-values for TD-updates (following Dead-ends criteria)
        ETA (float): The scaling factor used by the distortion risk measure
        BATCH_SIZE (int): size of the training batch
        BUFFER_SIZE (int): size of the replay memory
        LR (float): learning rate
        TAU (float): tau for soft updating the network weights
        GAMMA (float): discount factor
        UPDATE_EVERY (int): update frequency
        device (str): device that is used for the compute
        seed (int): random seed
    """

    def __init__(self, state_size, params, sided_Q='negative', device="cuda:0"):
        super(IQN_Agent, self).__init__()
        
        # Establish the parameters of learning the value function with this Network
        self.seed = torch.manual_seed(params.get("seed", 0))
        self.state_dim = state_size  # Dimension of the state observations
        self.action_size = params.get("num_actions", 25)  # Dimension of the action space (discrete)
        self.layer_size = params.get("num_q_hidden_units", 64)
        self.n_step = params.get("num_steps", 1)
        self.gamma = params.get("gamma", 1.0)
        self.lr = params.get("lr", 2.5e-4)
        self.tau = params.get("tau", 0.005)  # The weight of the Polyak average between the local and target networks
        self.device = device  # The device that models and data will be loaded to
        self.sided_Q = sided_Q  # Whether we'll be focusing on positive/negative rewards only
        
        # IQN Specific parameters
        self.N = params.get("num_iqn_samples_train", 32)  # The number of samples to draw from the value distributions when training
        self.K = params.get("num_iqn_samples_est", 32)  # The number of samples to draw from the value distribution when estimating the value function for the policy.
        self.drm = params.get("distortion_risk_measure", "identity")  # The distortion risk measure that will be used to transform sample distribution from Value functions; choices=['identity','cvar','cpw','power']
        self.eta = params.get("eta", 0.71)  # The scaling factor for the Distortion Risk Measure
        self.dueling = params.get("dueling", False)  # Whether we'll have a dueling DQN base for the IQN
        self.noisy = params.get("noisy_layer", False)  # Whether we'll implement a noisy linear layer

        self.use_cql = params.get("use_cql", False)
        self.min_z_weight = params.get("cql_weight", 0.0)
        

        # Establish the local and target network for Q-learning
        self.network = IQN(state_size, self.action_size, self.layer_size, self.n_step, 
                        N=self.N, K=self.K, drm=self.drm, eta=self.eta, use_cql=self.use_cql, 
                        dueling=self.dueling, noisy=self.noisy, device=self.device)
        self.target_network = IQN(state_size, self.action_size, self.layer_size, self.n_step, 
                        N=self.N, K=self.K, drm=self.drm, eta=self.eta, use_cql=self.use_cql, 
                        dueling=self.dueling, noisy=self.noisy, device=self.device)

        self.network.to(self.device)
        self.target_network.to(self.device)
        self.update_counter = 0
        self.update_freq = params.get("q_update_freq", 25)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        print(self.network)

        self.t_step = 0

    def _train_on_batch(self, s, a, r, s2, t):
        """With the provided batch of data, run a training update"""

        batch_size = s.shape[0]
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.target_network(s2, self.N)
        Q_targets_next = Q_targets_next.detach().cpu()
        action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)
        Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(batch_size, self.N, 1)).transpose(1, 2)

        # Compute Q targets for current states, clamp Q-values depending on what side of DeD we're using
        # Have to .unsqueeze() twice here to handle the correct sizing... (All action, reward and terminal vectors are single dimensional...)
        if self.sided_Q == 'negative':
            Q_targets = torch.clamp(r, max=0.0, min=-1.0).unsqueeze(-1).unsqueeze(-1) + (self.gamma**self.n_step * torch.clamp(Q_targets_next.to(self.device), max=0.0, min=-1.0) * (1 - t).unsqueeze(-1).unsqueeze(-1))
        elif self.sided_Q == 'positive':
            Q_targets = torch.clamp(r, max=1.0, min=0.0).unsqueeze(-1).unsqueeze(-1) + (self.gamma**self.n_step * torch.clamp(Q_targets_next.to(self.device), max=1.0, min=0.0) * (1 - t).unsqueeze(-1).unsqueeze(-1))
        else:
            Q_targets = torch.clamp(r, max=1.0, min=-1.0).unsqueeze(-1).unsqueeze(-1) + (self.gamma**self.n_step * torch.clamp(Q_targets_next.to(self.device), max=1.0, min=-1.0) * (1 - t).unsqueeze(-1).unsqueeze(-1))

        # Get expected Q values from local network
        Q_values, taus = self.network(s, self.N)
        Q_expected = Q_values.gather(2, a.unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.N, 1))

        # Quantile Huber Loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (batch_size, self.N, self.N), "wrong TD error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus.to(self.device) - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1)
        loss = loss.mean()

        if self.use_cql:

            # Sample the quantile that will be used for each datapoint
            # Expanding to be the shape of the Q-values
            penalty_index = torch.from_numpy(np.random.randint(0, self.N, size=batch_size)).unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, self.action_size).to(self.device)
            z_rand = Q_values.gather(1, penalty_index)
            
            # Compute the logsumexp of the Q-value distributions over actions
            min_zf_loss = torch.logsumexp(z_rand.squeeze(), -1).mean() # Average over the batch dimension

            # Compute the CQL penalty by subtracting `min_zf_loss` by the average expected Q-value of the observed actions
            # Weighted by the alpha parameter, static and fixed since we're doing Q-learning (as opposed to double gradient descent in Actor-critic formulations...)
            penalty = (min_zf_loss - Q_expected.squeeze().mean()) * self.min_z_weight

            # Adjust the loss according to the CQL/CODAC penalty
            loss = loss + penalty

        # Apply the gradient update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1)  # Clip the gradients...
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network)
        return loss.detach().cpu().numpy()


    def get_loss(self, s, a, r, s2, t):
        """Calculate the loss without gradients. (For diagnostic purposes/validation)"""

        batch_size = s.shape[0]

        with torch.no_grad():
            # Get max predicted Q values (for next states) from target model
            Q_targets_next, _ = self.target_network(s2, self.N)
            Q_targets_next = Q_targets_next.detach().cpu()
            action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)
            Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(batch_size, self.N, 1)).transpose(1, 2)

            # Compute Q targets for current states, clamp Q-values depending on what side of DeD we're using
            # Have to .unsqueeze() twice here to handle the correct sizing... (All action, reward and terminal vectors are single dimensional...)
            if self.sided_Q == 'negative':
                Q_targets = torch.clamp(r, max=0.0, min=-1.0).unsqueeze(-1).unsqueeze(-1) + (self.gamma**self.n_step * torch.clamp(Q_targets_next.to(self.device), max=0.0, min=-1.0) * (1 - t).unsqueeze(-1).unsqueeze(-1))
            elif self.sided_Q == 'positive':
                Q_targets = torch.clamp(r, max=1.0, min=0.0).unsqueeze(-1).unsqueeze(-1) + (self.gamma**self.n_step * torch.clamp(Q_targets_next.to(self.device), max=1.0, min=0.0) * (1 - t).unsqueeze(-1).unsqueeze(-1))
            else:
                Q_targets = torch.clamp(r, max=1.0, min=-1.0).unsqueeze(-1).unsqueeze(-1) + (self.gamma**self.n_step * torch.clamp(Q_targets_next.to(self.device), max=1.0, min=-1.0) * (1 - t).unsqueeze(-1).unsqueeze(-1))

            # Get expected Q values from local network
            Q_values, taus = self.network(s, self.N)
            Q_expected = Q_values.gather(2, a.unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.N, 1))

            # Quantile Huber Loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (batch_size, self.N, self.N), "wrong TD error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus.to(self.device) - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = quantil_l.sum(dim=1).mean(dim=1)
            loss = loss.mean()

            if self.use_cql:
                
                # Sample the quantile that will be used for each datapoint
                # Expanding to be the shape of the Q-values
                penalty_index = torch.from_numpy(np.random.randint(0, self.N, size=batch_size)).unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, self.action_size).to(self.device)
                z_rand = Q_values.gather(1, penalty_index)
                
                # Compute the logsumexp of the Q-value distributions over actions
                min_zf_loss = torch.logsumexp(z_rand.squeeze(), -1).mean() # Average over the batch dimension

                # Compute the CQL penalty by subtracting `min_zf_loss` by the average expected Q-value of the observed actions
                # Weighted by the alpha parameter, static and fixed since we're doing Q-learning (as opposed to double gradient descent in Actor-critic formulations...)
                penalty = (min_zf_loss - Q_expected.squeeze().mean()) * self.min_z_weight

                # Adjust the loss according to the CQL/CODAC penalty
                loss = loss + penalty

        return loss.detach().cpu().numpy()

    def learn(self, s, a, r, s2, term):
        """Learning from one minibatch."""
        loss = self._train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.soft_update(self.network, self.target_network)  # TODO Reconfigure for IQN...
            # self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1

        return loss

    def estimate_q_dist(self, input_states, num_samples, use_drm=False):
        """With the specified number of samples and with the input states, compute the value distributions over the actions"""
        q_values, _ = self.network(input_states, num_samples, use_drm)
        return q_values
        
    def dump_network(self, weights_file_path):
        try:
            torch.save(self.network.state_dict(), weights_file_path)
        except:
            pass

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
    
    def resume(self, network_state_dict, target_network_state_dict, optimizer_state_dict):
        self.network.load_state_dict(network_state_dict)
        self.target_network.load_state_dict(target_network_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters (Polyak averaging).
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    @staticmethod
    def weight_transfer(self, from_model, to_model):
        """Directly copies the local network parameters to the target network."""
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # Not pickle-able
        return _dict

##########################################################
#           AGENT DEFINITIONS (DDQN and IQN)
##########################################################

def trainer(Dnet, Rnet, train_loader, num_epochs, dtype=torch.float, device="cpu"):

    # Set models to train mode (we're going to be handling them one at a time...)
    Dnet.to(dtype=dtype, device=device)
    Dnet.train()
    Rnet.to(dtype=dtype, device=device)
    Rnet.train()

    # Optimizer is defined internal to the Agent definitions...

    # Reset the DataLoaders, primes them for training
    train_loader.reset(shuffle=True)

    # Loop through the training DataLoader
    print(f"Training the D- and R- networks for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Cycle through the dataloader
        epoch_done = False
        epoch_steps = 0
        epoch_loss = 0
        while not epoch_done:
            states, actions, rewards, next_states, terminals, epoch_done = train_loader.get_next_minibatch()
            epoch_steps += len(states)
            loss = Dnet.learn(states, actions, rewards, next_states, terminals)
            loss = Rnet.learn(states, actions, rewards, next_states, terminals)
            epoch_loss += loss
        
        # Reset the training DataLoader
        train_loader.reset(shuffle=True)

    return Dnet, Rnet

def evaluator(Dnet, Rnet, val_data, thresholds, distributional=True, device="cpu", output_type="full"):
    """
    Evaluate the trained D- and R- networks, using the provided validation data.

    We calculate the Q-values for all state x action pairs for all possible settings
    of the CVaR thresholds.
    """
    print("Evaluating the trained D- and R- Networks")
    # Set networks to eval()
    Dnet.eval()
    Rnet.eval()


    # Calculate the value distributions of all states in each trajectory
    data = get_dn_rn_info(Dnet, Rnet, val_data, device, distributional=distributional)
    # Split out the computed data based on the treatments chosen and patient outcome 
    results = pre_flag_splitting(data, thresholds, distributional=distributional)

    # Extract the number of surviving and non-surviving patients
    num_survivors = len(results['survivors']['dn_q_selected_action_traj'])
    num_nonsurvivors = len(results['nonsurvivors']['dn_q_selected_action_traj'])

    # Create the analysis DataFrames to be used for computation of particular patient risk profiles
    surv_df, nonsurv_df = create_analysis_df(results, num_survivors, num_nonsurvivors)

    # Calculate the AUC from the analysis DataFrames (We don't care about the TPR/FPR arrays at this stage)
    if not distributional:
        fpr, tpr, out_auc = compute_auc(surv_df, nonsurv_df, num_survivors, num_nonsurvivors)  # Need 'iqn_size' to revert to the default 'None'
    else:
        fpr, tpr, out_auc = compute_auc(surv_df, nonsurv_df, num_survivors, num_nonsurvivors, iqn_size=len(thresholds))
    

    if output_type == 'full':
        print(f'Max AUC for this model: {np.max(out_auc)}')
        return fpr, tpr, out_auc, data, results
    elif output_type == 'mean':
        print(f'Average AUC for this model: {np.mean(out_auc)}')
        return np.mean(out_auc)