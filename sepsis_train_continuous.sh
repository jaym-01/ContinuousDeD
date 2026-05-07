python3 ncde_train.py -c ncde_config_continuous

python3 encode_data.py -c ncde_config_continuous

python3 train_rl.py -c iqn_continuous_mimic

python3 eval_rl.py -c iqn_continuous_mimic