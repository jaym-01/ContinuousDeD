## IMPORTS
import os, sys, time
import json
import random
import pickle
import click
import yaml
import numpy as np

import torch

from rl_utils import RLDataLoader, DQN_Agent, IQN_Agent, ContinuousIQN_OfflineAgent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################
##     HELPER FUNCTIONS
#############################

def train_network(params, device, train_loader, val_loader):
    """Using the `params`, define a Q-Network to be trained with `train_loader` and periodically validated using `val_loader`."""

    # Smoke-test mode: trim data to 500 transitions and cap at 2 epochs
    if params.get('smoke_test', False):
        print(">>> SMOKE TEST MODE: capping at 500 transitions and 2 epochs <<<")
        for key in train_loader.transition_data:
            train_loader.transition_data[key] = dict(
                list(train_loader.transition_data[key].items())[:500]
            )
        train_loader.transition_data_size = 500
        train_loader.transition_indices = np.arange(500)
        train_loader.transition_indices_pos_last = [
            i for i in train_loader.transition_indices_pos_last if i < 500
        ]
        train_loader.transition_indices_neg_last = [
            i for i in train_loader.transition_indices_neg_last if i < 500
        ]
        params['num_epochs'] = 2

    # Initialize the model
    if params['model'] == 'DQN':
        model = DQN_Agent(params['input_dim'], params, sided_Q=params['sided_Q'], device=device)
    elif params['model'] == 'IQN':
        model = IQN_Agent(params['input_dim'], params, sided_Q=params['sided_Q'], device=device)
    elif params['model'] == 'ContinuousIQN':
        model = ContinuousIQN_OfflineAgent(params['input_dim'], params, sided_Q=params['sided_Q'], device=device)
    else:
        raise NotImplementedError('The provided model type has not yet been defined, please use DQN, IQN, or ContinuousIQN')

    # Set-up train/val artifacts
    curr_epoch = 0
    all_epoch_steps = []
    all_epoch_validation_steps = []
    all_epoch_loss = []
    all_epoch_validation_loss = []

    best_val_loss = 1e6

    # Reset the DataLoaders
    train_loader.reset(shuffle=True)
    val_loader.reset(shuffle=False)

    # Loop through the training DataLoader
    for epoch in range(curr_epoch, params['num_epochs']):
        print()
        print(f">>>>>>>>> Experiment Q-{params['sided_Q'].capitalize()}: Epoch {epoch+1} of {params['num_epochs']}")
        # Cycle through the training dataloader
        epoch_done = False
        epoch_steps = 0
        epoch_loss = 0
        while not epoch_done:
            states, actions, rewards, next_states, terminals, epoch_done = train_loader.get_next_minibatch()
            epoch_steps += len(states)
            loss = model.learn(states, actions, rewards, next_states, terminals)
            epoch_loss += loss

        train_loader.reset(shuffle=True)
        val_loader.reset(shuffle=False)
        all_epoch_loss.append(epoch_loss/epoch_steps)
        all_epoch_steps.append(epoch_steps)
        # Periodically, validate the trained network using the validation DataLoader and save off checkpoint 
        if (epoch+1) % params['saving_period'] == 0:
            val_loss, val_steps = validate_network(model, val_loader)
            all_epoch_validation_loss.append(val_loss/val_steps)
            all_epoch_validation_steps.append(val_steps)
            try:
                torch.save({
                    'epoch': epoch,
                    'rl_network_state_dict': model.network.state_dict(),
                    'loss': all_epoch_loss,
                    'validation_loss': all_epoch_validation_loss,
                    'epoch_steps': all_epoch_steps,
                    'epoch_validation_steps': all_epoch_validation_steps
                }, os.path.join(params['checkpoint_fname'], f"q_parameters{params['sided_Q']}.pt"))
                np.save(os.path.join(params['checkpoint_fname'], f"q_losses_{params['sided_Q']}"), all_epoch_loss)
            except:
                print(">>> Cannot save...")
                

            # Save the best model, based on Validation Loss
            if val_loss/val_steps <= best_val_loss:
                best_val_loss = val_loss / val_steps
                print(f"New Best Validation Loss: {best_val_loss}")
                try:
                    torch.save({
                        'epoch': epoch,
                        'rl_network_state_dict': model.network.state_dict(),
                        'loss': all_epoch_loss,
                        'validation_loss': all_epoch_validation_loss,
                    }, os.path.join(params['checkpoint_fname'], f"best_q_parameters{params['sided_Q']}.pt"))
                except:
                    print(">>> Cannot save...")


def validate_network(model, val_loader):
    # Set up artifacts
    val_steps = 0
    val_loss = 0
    epoch_done = False
    # Loop through the validation DataLoader
    while not epoch_done:
        states, actions, rewards, next_states, terminals, epoch_done = val_loader.get_next_minibatch()
        val_steps += len(states)
        loss = model.get_loss(states, actions, rewards, next_states, terminals)
        val_loss += loss
    
    
    return val_loss, val_steps



#############################
##     MAIN EXECUTOR
#############################

@click.command()
@click.option('--config', '-c', default='dqn_baseline')
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(config, options):
    """Using the specified configuration, train D and R networks (if performing DeD) with RL"""
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, f'configs/{config}.yaml')))

    # Replace configuration parameteres by command line provided 'options'
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

    combined = params.get("combined", False)
    dataset = params.get("dataset", "train")

    # Ensure that the desired output directory exists, if not create it...
    if not os.path.exists(params['checkpoint_fname']):
        os.mkdir(params['checkpoint_fname'])

    # Initialize the data-loaders from the NCDE encoded data
    train_loader = RLDataLoader(
                    params['data_dir'], rng, params['train_batch_size'],
                    dataset=dataset, pos_samples_in_minibatch=params['num_ps'],
                    neg_samples_in_minibatch=params['num_ns'], device=device)
    # Build transitions without releasing — state stats needed for continuous agent
    train_loader.make_transition_data(release=False)

    meta_path = os.path.join(params['data_dir'], "encoded_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        ncde_hidden_dim = meta.get("ncde_hidden_dim", None)
        if ncde_hidden_dim is not None and int(ncde_hidden_dim) != int(train_loader.data_dim):
            raise ValueError(
                "Encoded data dimension does not match NCDE metadata. "
                f"encoded state_dim={train_loader.data_dim}, ncde_hidden_dim={ncde_hidden_dim}. "
                "Re-run encode_data.py with the intended NCDE model or point RL to the correct data_dir."
            )
        if train_loader.continuous_actions:
            enc_action_dim = int(train_loader.encoded_actions.shape[-1])
            cfg_action_dim = int(params.get("action_dim", enc_action_dim))
            if enc_action_dim != cfg_action_dim:
                raise ValueError(
                    "Encoded action_dim does not match RL config. "
                    f"encoded action_dim={enc_action_dim}, config action_dim={cfg_action_dim}. "
                    "Update iqn config or regenerate encoded data."
                )
    else:
        print(
            "Warning: encoded_meta.json not found; cannot verify NCDE/RL dimensional consistency. "
            "Re-run encode_data.py to generate metadata."
        )

    params['input_dim'] = train_loader.data_dim

    # For the continuous IQN, compute per-dim state mean/std for normalisation
    if params.get('model') == 'ContinuousIQN':
        print("Computing state normalisation statistics from training data...")
        state_mean, state_std = train_loader.compute_state_stats()
        params['state_mean'] = state_mean
        params['state_std']  = state_std

    # Now release the raw encoded arrays to free memory
    train_loader.release()

    val_loader = RLDataLoader(
                    params['data_dir'], rng, params['val_batch_size'],
                    dataset='validation', device=device)
    val_loader.make_transition_data(release=True)


    
    if combined:
        addon = " with CQL penalty" if params['use_cql'] else ""
        msg = f"Training {config[:3].upper()} with all (positive and negative) rewards" + addon
        print(msg)
        params['sided_Q'] = 'both'
        train_network(params, device, train_loader, val_loader)
    else: # Loop over the constraint of positive/negative rewards when training the RL Networks
        for sided_Q in ['negative', 'positive']:
            addon = " with CQL penalty" if params['use_cql'] else ""
            msg = f"Training {config[:3].upper()} with only {sided_Q.capitalize()} rewards" + addon
            print(msg)
            params['sided_Q'] = sided_Q
            train_network(params, device, train_loader, val_loader)



if __name__ == '__main__':
    run()
