"""
finetune_rl.py

This script is in place to finetune the hyperparameters to define and train
the IQN+CQL agent using the Ax BayesOpt package.
"""
# IMPORTS
import os, sys, time
import yaml
import random
import numpy as np
import click

import torch

from ax.service.managed_loop import optimize

from rl_utils import RLDataLoader, IQN_Agent, DQN_Agent, trainer, evaluator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

@click.command()
@click.option('--config', '-c', default='rl_finetuning_config')
def run(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration file
    config_dict = yaml.safe_load(open(f"./configs/{config}.yaml", "r"))
    data_dir = config_dict['data_dir']

    # Set the random seeds
    torch.manual_seed(config_dict['seed'])
    random.seed(config_dict['seed'])
    rng = np.random.RandomState(config_dict['seed'])

    if device == "cuda":  # Adjust the random seed on the cuda device
        torch.cuda.manual_seed(config_dict['seed'])

    # Load the data
    if not os.path.exists(config_dict['output_dir']):
        os.mkdir(config_dict['output_dir'])

    # Initialize the data-loaders from the NCDE encoded data
    train_loader = RLDataLoader(
                    config_dict['data_dir'], rng, config_dict['train_batch_size'], 
                    dataset='train', pos_samples_in_minibatch=config_dict['num_ps'], 
                    neg_samples_in_minibatch=config_dict['num_ns'], device=device)
    train_loader.make_transition_data(release=True)

    config_dict['input_dim'] = train_loader.data_dim

    # Load the validation data
    validation_data = np.load(os.path.join(data_dir, 'encoded_validation.npz'), allow_pickle=True)

    # Define the VaR_thresholds
    VaR_thresholds = np.round(np.linspace(0.05, 1.0, num=20), decimals=2)

    # Define the training function that will be used by Ax to optimize the hyperpameters
    def train_evaluate(parameterization):
        """Helper function that iniailizes the model to be trained and sets up the training module"""

        
        if 'iqn' in config_dict['architecture']:
            dist_arch = True
            # Initialize the IQN based agent
            model_d = IQN_Agent(config_dict['input_dim'], parameterization, sided_Q='negative', device=device)
            model_d = model_d.to(device)

            model_r = IQN_Agent(config_dict['input_dim'], parameterization, sided_Q='positive', device=device)
            model_r = model_r.to(device)
        elif 'dqn' in config_dict['architecture']:
            dist_arch = False
            # Intiialize the DQN based agent
            model_d = DQN_Agent(config_dict['input_dim'], parameterization, sided_Q='negative', device=device)
            model_d = model_d.to(device)

            model_r = DQN_Agent(config_dict['input_dim'], parameterization, sided_Q='positive', device=device)
            model_r = model_r.to(device)
        else:
            raise NotImplementedError('The provided model type has not yet been defined, please use DQN or IQN')


        num_epochs = parameterization.get("num_training_epochs", 100)

        # Run the training loop
        model_d, model_r = trainer(Dnet=model_d, Rnet=model_r, train_loader=train_loader, num_epochs=num_epochs, dtype=torch.float, device=device)

        return evaluator(Dnet=model_d, Rnet=model_r, val_data=validation_data, thresholds=VaR_thresholds, distributional=dist_arch, device=device, output_type='mean')


    # Set up and run the optimization loop with Ax
    best_parameters, values, experiment, model = optimize(
            parameters = config_dict['parameterization'],
            evaluation_function=train_evaluate,
            objective_name = 'AUC',
            total_trials = config_dict['total_trials'],
            minimize=False,
    )

    # Using the best set of parameters, reset the model and re-train using the combined training+validation set
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    print("Best arm: ", best_arm)

    # Initialize the model
    if 'iqn' in config_dict['architecture']:
        model_d = IQN_Agent(config_dict['input_dim'], best_arm.parameters, sided_Q='negative', device=device)
        model_d = model_d.to(device)
        model_r = IQN_Agent(config_dict['input_dim'], best_arm.parameters, sided_Q='positive', device=device)
        model_r = model_r.to(device)
    else:
        model_d = DQN_Agent(config_dict['input_dim'], best_arm.parameters, sided_Q='negative', device=device)
        model_d = model_d.to(device)
        model_r = DQN_Agent(config_dict['input_dim'], best_arm.parameters, sided_Q='positive', device=device)
        model_r = model_r.to(device)

    # Re-initialize a combined train-val dataloader
    train_val_loader = RLDataLoader(
                    config_dict['data_dir'], rng, config_dict['train_batch_size'], 
                    dataset='train_val', pos_samples_in_minibatch=config_dict['num_ps'], 
                    neg_samples_in_minibatch=config_dict['num_ns'], device=device)
    train_val_loader.make_transition_data(release=True)

    num_epochs = best_arm.parameters.get("num_training_epochs", 100)

    # Run training loop
    model_d, model_r = trainer(Dnet=model_d, Rnet=model_r, train_loader=train_val_loader, num_epochs=num_epochs, dtype=torch.float, device=device)

    # Save off the model parameters!
    save_dict = {
        'Dnetwork': model_d.network.state_dict(),
        'Rnetwork': model_r.network.state_dict(),
        'hyperparameters': best_arm.parameters,
        'experiment_data': data,
    }
    torch.save(save_dict, config_dict['output_dir']+config_dict['output_fname'])


if __name__ == '__main__':
    run()
    
