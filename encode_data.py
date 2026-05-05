"""
Encode all the data using the best performing NCDE model.
"""
import os, sys, time
import argparse
import yaml
import random
import numpy as np

import torch

from ncde_utils import load_data
from ncde import NeuralCDE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='ncde_config', help='Config name (without .yaml) in ./configs/')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load the configuration file
    config_dict = yaml.safe_load(open(f"./configs/{args.config}.yaml", 'r'))

    # Set the random seeds
    torch.manual_seed(config_dict['seed'])
    random.seed(config_dict['seed'])
    if device == "cuda":
        torch.cuda.manual_seed(config_dict['seed'])

    action_kwargs = {
        'one_hot_actions': config_dict.get('one_hot_actions', False),
        'n_action_bins':   config_dict.get('n_action_bins', None),
    }
    encode_output_dir = config_dict.get('encode_output_dir', config_dict['data_dir'])
    os.makedirs(encode_output_dir, exist_ok=True)

    # Load the data (both the nominal and overlapping data)
    dl_types = ("train", "validation", "test")
    dataloaders, input_dim, action_dim, static_dim, output_dim = load_data(
        data_dir=config_dict['data_dir'], batch_size=config_dict['batch_size'], **action_kwargs
    )
    (overlap_dataloader, __, __), _, _, _, _ = load_data(
        data_dir=config_dict['data_dir'], batch_size=config_dict['batch_size'], overlap=True, **action_kwargs
    )

    # Load the best performing NCDE hyperparameters and pre-trained model weights
    ncde_params = torch.load(os.path.join(config_dict['output_dir'], 'best_model.pt'))
    ncde_hparams = ncde_params['hyperparameters']

    hdim = ncde_hparams['hidden_dim']
    hhdim = ncde_hparams['hidden_hidden_dim']
    p_n_layers = ncde_hparams['pred_num_layers']
    p_n_units = ncde_hparams['pred_num_units']
    
    # Initialize the model
    model = NeuralCDE(input_dim, hdim, output_dim, static_dim, action_dim,
                        hidden_hidden_dim=hhdim, pred_num_layers=p_n_layers,
                        pred_num_units=p_n_units, return_sequences=True, device=device)
    
    # Load the pre-trained model parameters, set to evaluation mode
    model.load_state_dict(ncde_params['model'])
    model = model.to(device)

    model.eval()

    # Now loop through the dataloaders to encode the time-series data, package up and store the encoded representations with the corresponding intervention and outcome information
    for i, (dataset, dl) in enumerate(zip(dl_types, dataloaders)):
        encoded = []
        actions = []
        rewards = []
        lengths = []
        stay_ids = []
        with torch.no_grad():
            for ii, (inputs, _, lens, _, outcomes, sids) in enumerate(dl):
                static, temporal, acts = inputs
                static = static.to(device)
                temporal = temporal.to(device)
                acts = acts.to(device)

                inputs = (static, temporal, acts)

                __, hidden = model(inputs)

                # Store the batches of data
                encoded.append(hidden.cpu().numpy())
                actions.append(acts.cpu().numpy())
                rewards.append(outcomes.numpy())
                lengths.extend(lens.numpy().ravel())  # Saving off the lengths because we'll want to remove all the padding when constructing our RL dataset
                stay_ids.extend(sids.numpy()) # Saving off the individual trajectory stay IDs for analysis purposes...


            # Remove the odd grouping from appending to lists...
            encoded = np.vstack(encoded)
            actions = np.vstack(actions)
            rewards = np.vstack(rewards)
            lengths = np.vstack(lengths)

            # Save off an npz file for this part of the dataset
            np.savez(
                os.path.join(encode_output_dir, "encoded_{}.npz".format(dataset)),
                states=encoded,
                actions=actions,
                rewards=rewards,
                lengths=lengths,
                stay_ids=stay_ids,
            )

    # Loop over the overlap data and encode it
    encoded = []
    actions = []
    rewards = []
    lengths = []
    stay_ids = []
    with torch.no_grad():
        for ii, (inputs, _, lens, outcomes, sids) in enumerate(overlap_dataloader):
            static, temporal, acts = inputs
            static = static.to(device)
            temporal = temporal.to(device)
            acts = acts.to(device)

            inputs = (static, temporal, acts)

            __, hidden = model(inputs)

            # Store the batches of data
            encoded.append(hidden.cpu().numpy())
            actions.append(acts.cpu().numpy())
            rewards.append(outcomes.numpy())
            lengths.extend(lens.numpy())  # Saving off the lengths because we'll want to remove all the padding when constructing our RL dataset
            stay_ids.extend(sids.numpy()) # Saving off the individual trajectory stay IDs for analysis purposes...

        # Remove the odd grouping from appending to lists...
        encoded = np.vstack(encoded)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        lengths = np.vstack(lengths)

        # Save off an npz file for this part of the dataset
        np.savez(
            os.path.join(encode_output_dir, "encoded_overlap.npz"),
            states=encoded,
            actions=actions,
            rewards=rewards,
            lengths=lengths,
            stay_ids=stay_ids,
        )