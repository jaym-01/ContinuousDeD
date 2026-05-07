"""
This file is mostly in place to preprocess the data and save it off for use later! That way we can avoid
the overhead of continually running the rectilinear interpolation procedure over and over...
"""
import os, sys, time
import argparse
import yaml
import random
import numpy as np

import torch

from ax.service.managed_loop import optimize

from ncde_utils import process_interpolate_and_save, load_data, trainer, evaluator
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

    if device == "cuda":  # Adjust the random seed on the cuda device
        torch.cuda.manual_seed(config_dict['seed'])

    # Adjust the config based on possible command line inputs... TODO...

    # Ensure output directory exists
    os.makedirs(config_dict['output_dir'], exist_ok=True)

    action_kwargs = {
        'one_hot_actions': config_dict.get('one_hot_actions', False),
        'n_action_bins':   config_dict.get('n_action_bins', None),
    }

    # Load the data
    (train_loader, val_loader, test_loader), input_dim, action_dim, static_dim, output_dim = load_data(
        data_dir=config_dict['data_dir'], batch_size=config_dict['batch_size'], **action_kwargs
    )


    # Define the training function that will be used by Ax to optimize the hyperparameters
    def train_evaluate(parameterization):
        """Helper function that initializes the model to be trained and sets up the training module"""

        # Extract model specific parameterizations
        hidden_dim = parameterization.get("hidden_dim", 128)
        hidden_hidden_dim = parameterization.get("hidden_hidden_dim", 16)
        pred_num_layers = parameterization.get("pred_num_layers", 2)
        pred_num_units = parameterization.get("pred_num_units", 100)

        # Initialize the model
        model = NeuralCDE(input_dim, hidden_dim, output_dim, static_dim, action_dim, 
                            hidden_hidden_dim=hidden_hidden_dim, pred_num_layers=pred_num_layers, 
                            pred_num_units=pred_num_units, return_sequences=True, device=device)
        model = model.to(device)

        # Run training loop
        model = trainer(model=model, train_loader=train_loader, parameters=parameterization, dtype=torch.float, device=device)

        return evaluator(model=model, eval_loader=val_loader, dtype=torch.float, device=device)


    # Set up and run the optimization loop with Ax
    best_parameters, values, experiment, model = optimize(
        parameters = config_dict['parameterization'],
        evaluation_function = train_evaluate,
        objective_name = 'Masked_MSE',
        total_trials = config_dict['total_trials'],
        minimize = True,
    )

    # Using the best set of parameters, reset the model and re-train using the combined training+validation set
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].min()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    print("Best arm: ", best_arm)

    # Create a combined Train/Val dataset!
    (combined_train_valid_loader, __, test_loader), __, __, __, __ = load_data(
        data_dir=config_dict['data_dir'], batch_size=config_dict['batch_size'],
        combine_train_val=True, shuffle=True, **action_kwargs
    )

    ## Re-initialize the model using the best parameters
    ################
    # Extract model specific parameterizations
    hidden_dim = best_arm.parameters.get("hidden_dim", 128)
    hidden_hidden_dim = best_arm.parameters.get("hidden_hidden_dim", 16)
    pred_num_layers = best_arm.parameters.get("pred_num_layers", 2)
    pred_num_units = best_arm.parameters.get("pred_num_units", 100)

    # Initialize the model
    model = NeuralCDE(input_dim, hidden_dim, output_dim, static_dim, action_dim, 
                        hidden_hidden_dim=hidden_hidden_dim, pred_num_layers=pred_num_layers, 
                        pred_num_units=pred_num_units, return_sequences=True, device=device)
    model = model.to(device)

    # Run training loop
    model = trainer(model=model, train_loader=combined_train_valid_loader, parameters=best_arm.parameters, dtype=torch.float, device=device)

    # Save off the model parameters!
    save_dict = {
        'model': model.state_dict(),
        'hyperparameters': best_arm.parameters,
        'experiment_data': data,
    }
    torch.save(save_dict, os.path.join(config_dict['output_dir'], "best_model.pt"))

    # load_data(
    # data_dir="/ais/bulbasaur/twkillian/AHE_Sepsis_Data/rectilinear_processed/", 
    # use_static=True, overlap=False, 
    # batch_size=None, one_hot_actions=False, 
    # num_actions=None
    # ):

    # return (
    #     dataloaders,
    #     input_dim,
    #     action_dim,
    #     static_dim,
    #     output_dim
    # )

