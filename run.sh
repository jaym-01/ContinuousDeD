# Coarse grid — 5×5 = 25 actions

# .venv/bin/python toy_domain/run.py -agent iqn -action_mode discrete -n_bins 5 -ded -frames 100000 -info ded_discrete_5bins

# # Finer grid — 9×9 = 81 actions
# .venv/bin/python toy_domain/run.py -agent iqn -action_mode discrete -n_bins 9 -ded -frames 500000 -info ded_discrete_9bins

# # Very fine — 17×17 = 289 actions (stress-test)
# .venv/bin/python toy_domain/run.py -agent iqn -action_mode discrete -n_bins 17 -ded -frames 500000 -info ded_discrete_17bins


# .venv/bin/python toy_domain/run.py -action_mode continuous -agent iqn -ded -info continuous_iqn_2 -alpha 0.2 -frames 50000 -anchor_ratio 0.7 -n_step 200

# LifeGate discrete
.venv/bin/python toy_domain/run.py -env LifeGate -action_mode discrete -agent iqn -ded -frames 10000 -info lifegate_iqn