## IMPORTS
import os, sys, time
import random
import pickle
import click
import yaml
import numpy as np

import torch

from rl_utils import RLDataLoader, DQN_Agent, IQN_Agent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################
##     HELPER FUNCTIONS
#############################

def train_network(params, device, train_loader, val_loader):
    """Using the `params`, define a Q-Network to be trained with `train_loader` and periodically validated using `val_loader`."""

    # Initialize the model
    if params['model'] == 'DQN':
        model = DQN_Agent(params['input_dim'], params, sided_Q=params['sided_Q'], device=device)
    elif params['model'] == 'IQN':
        model = IQN_Agent(params['input_dim'], params, sided_Q=params['sided_Q'], device=device)
    else:
        raise NotImplementedError('The provided model type has not yet been defined, please use DQN or IQN')

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
    train_loader.make_transition_data(release=True)

    params['input_dim'] = train_loader.data_dim

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
