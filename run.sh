cd toy_domain

# Coarse grid — 5×5 = 25 actions
../.venv/bin/python run.py -agent iqn -action_mode discrete -n_bins 5 -ded -frames 100000 -info ded_discrete_5bins

# # Finer grid — 9×9 = 81 actions
# ../.venv/bin/python run.py -agent iqn -action_mode discrete -n_bins 9 -ded -frames 500000 -info ded_discrete_9bins

# # Very fine — 17×17 = 289 actions (stress-test)
# ../.venv/bin/python run.py -agent iqn -action_mode discrete -n_bins 17 -ded -frames 500000 -info ded_discrete_17bins
