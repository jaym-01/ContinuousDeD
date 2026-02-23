# IMPORTS
import os, sys, time, re
import yaml
import random
import numpy as np
import click
import pickle

import torch

import pdb

from rl_utils import RLDataLoader, IQN_Agent, DQN_Agent, trainer, evaluator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=4)

data_dir = "/ais/bulbasaur/twkillian/AHE_Sepsis_Data/rectilinear_processed/"
output_dir = "/ais/bulbasaur/twkillian/UncDeD_Results/IQN/"

""" TO EVALUATE THE DIFFERENT SETTINGS OF CQL PENALTY...

Need to adjust the output directory that loads the `save_dict` to point to the various directories within ~/IQN/iqn_cql_wt_p*
Need to then load the parameters for both the negative and positive agents... 
This will then define the `model_d` and `model_r` used for evaluating the appraoch... We can hopefully set this up in a loop?

"""

@click.command()
@click.option('--model', '-m', default='iqn_cql')
@click.option('--data', '-d', default='test')
def run(model, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = f'best_{model}_model.pt'
    

    if 'dqn' in model:
        output_dir = "/ais/bulbasaur/twkillian/UncDeD_Results/DQN/"
    else:
        output_dir = "/ais/bulbasaur/twkillian/UncDeD_Results/IQN/"

    addon = "_overlap" if data=='overlap' else ""
    
    # Load the trained model weights and parameters
    save_dict = torch.load(os.path.join(output_dir, model, model_file))

    # negative_params = os.path.join(f'{model}', 'best_q_parametersnegative.pt')
    # positive_params = os.path.join(f'{model}', 'best_q_parameterspositive.pt')
    # negative_model_params = torch.load(os.path.join(output_dir, negative_params))
    # positive_model_params = torch.load(os.path.join(output_dir, positive_params))

    params_dict = save_dict['hyperparameters']

    # params_dict = {'cql_weight': int(re.findall(r'\d', model)[0])/10, 
    #                 'num_iqn_samples_train': 128, 'num_iqn_samples_est': 64, 
    #                 'num_q_hidden_units': 64, 'num_q_layers': 2, 'lr': 1e-05, 
    #                 'num_training_epochs': 75, 'seed': 0, 'num_actions': 25, 
    #                 'gamma': 1.0, 'distortion_risk_measure': 'identity', 
    #                 'q_update_freq': 5, 'use_cql': True}

    print(params_dict)

    if 'dqn' in model:
        params_dict['num_q_layers'] = 1

    # Set the random seeds
    torch.manual_seed(params_dict['seed'])
    random.seed(params_dict['seed'])
    rng = np.random.RandomState(params_dict['seed'])

    # torch.manual_seed(2022)
    # random.seed(2022)
    # rng = np.random.RandomState(2022)

    # Load the test data
    test_data = np.load(os.path.join(data_dir, f'encoded_{data}.npz'), allow_pickle=True)

    params_dict['input_dim'] = test_data['states'].shape[-1]

    # Initialize the D- and R- Networks
    if 'iqn' in model:
        dist_arch = True
        # Initialize the IQN based agent
        model_d = IQN_Agent(params_dict['input_dim'], params_dict, sided_Q='negative', device=device)
        model_d = model_d.to(device)
        model_d.network.load_state_dict(save_dict['Dnetwork'])
        # model_d.network.load_state_dict(negative_model_params['rl_network_state_dict'])
        model_d.eval()

        model_r = IQN_Agent(params_dict['input_dim'], params_dict, sided_Q='positive', device=device)
        model_r = model_r.to(device)
        model_r.network.load_state_dict(save_dict['Rnetwork'])
        # model_r.network.load_state_dict(positive_model_params['rl_network_state_dict'])
        model_r.eval()
    elif 'dqn' in model:
        dist_arch = False
        # Intiialize the DQN based agent
        model_d = DQN_Agent(params_dict['input_dim'], params_dict, sided_Q='negative', device=device)
        model_d = model_d.to(device)
        model_d.network.load_state_dict(negative_model_params['rl_network_state_dict'])
        model_d.eval()

        model_r = DQN_Agent(params_dict['input_dim'], params_dict, sided_Q='positive', device=device)
        model_r = model_r.to(device)
        model_r.network.load_state_dict(positive_model_params['rl_network_state_dict'])
        model_r.eval()
    else:
        raise NotImplementedError('The provided model type has not yet been defined, please use DQN or IQN')

    VaR_thresholds = np.round(np.linspace(0.0, 1.0, num=50), decimals=2)

    fpr, tpr, out_auc, data, results = evaluator(model_d, model_r, test_data, VaR_thresholds, distributional=('iqn' in model), device=device, output_type="full")

    # Save off the various items
    # results_dict = {
    #     'fpr': fpr,
    #     'tpr': tpr,
    #     'auc': out_auc,
    #     'data': data,
    #     'results': results
    # }
    results_dict = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': out_auc,
    }
    output_fname = f'best_{model}{addon}_auc_outputs.pkl'
    with open(os.path.join(output_dir, model, output_fname), 'wb') as f:
        pickle.dump(results_dict,f)

    output_fname = f'best_{model}{addon}_value_data.pkl'
    with open(os.path.join(output_dir, model, output_fname), 'wb') as f:
        pickle.dump(data, f)

    output_fname = f'best_{model}{addon}_pre_flag_results.pkl'
    with open(os.path.join(output_dir, model, output_fname), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    run()


"""IQN+CQL FT RESULTS

Max AUC for this model: 0.7978718872235778
--Return--
> /h/227/twkillian/Research/uncertainty-ded/eval_ft.py(75)run()->None
-> fpr, tpr, out_auc, data, results = evaluator(model_d, model_r, test_data, VaR_thresholds, distributional=('iqn' in model), device=device, output_type="full")
(Pdb) out_auc.shape
*** AttributeError: 'list' object has no attribute 'shape'
(Pdb) len(out_auc)
1
(Pdb) out_auc
[array([0.782 , 0.7852, 0.786 , 0.787 , 0.7879, 0.7887, 0.7897, 0.7907, 0.7916, 0.7923, 0.7931, 0.7939, 0.7946, 0.7952, 0.7958, 0.7962, 0.7965, 0.7969, 0.7974, 0.7979])]
(Pdb) np.mean(out_auc)
0.7919246930792766
(Pdb) np.mean(out_auc[0])
0.7919246930792766
(Pdb) np.min(out_auc[0])
0.7820165756554743
(Pdb) np.max(out_auc[0])
0.7978718872235778
"""

"""IQN+CQL(wt=0.1)
Max AUC for this model: 0.7422659850509277
"""

"""IQN+CQL(wt=0.2)
Max AUC for this model: 0.6822547538824987
"""

"""IQN+CQL(wt=0.3)
Max AUC for this model: 0.5728999651446497
"""

"""IQN+CQL(wt=0.4)
Max AUC for this model: 0.5020797025676774
"""


"""IQN FT RESULTS
(Pdb) out_auc
[array([0.7569, 0.7656, 0.7689, 0.771 , 0.7727, 0.7739, 0.7748, 0.7756, 0.7764, 0.777 , 0.7775, 0.7779, 0.7783, 0.7786, 0.779 , 0.7793, 0.7796, 0.7798, 0.7798, 0.7797])]
(Pdb) np.mean(out_auc)
0.7750974981604121
(Pdb) np.min(out_auc[0])
0.7568742496417644
(Pdb) np.max(out_auc[0])
0.7798110065450602
(Pdb) np.mean(out_auc[0])
0.7750974981604121
"""

"""DQN+CQL RESULTS
Max AUC for this model: 0.47311490647147675
> /h/227/twkillian/Research/uncertainty-ded/eval_ft.py(80)run()
-> 'fpr': fpr,
(Pdb) out_auc
[0.47311490647147675]

Best DQN+CQL params... --> 0.8122 AUC (!!)
cql_wt 0.25, num_q_hidden_units 16
num_q_layers: 2, lr: 7.825175814457108e-5
num_training_epochs: 75, seed: 0,
num_actions: 25, gamma: 1.0
distortion_risk_measure: 'identity', q_update_freq: 5,
use_cql: True


"""