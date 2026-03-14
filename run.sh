# Coarse grid — 5×5 = 25 actions

# .venv/bin/python toy_domain/run.py -agent iqn -action_mode discrete -n_bins 5 -ded -frames 100000 -info ded_discrete_5bins

# # Finer grid — 9×9 = 81 actions
# .venv/bin/python toy_domain/run.py -agent iqn -action_mode discrete -n_bins 9 -ded -frames 500000 -info ded_discrete_9bins

# # Very fine — 17×17 = 289 actions (stress-test)
# .venv/bin/python toy_domain/run.py -agent iqn -action_mode discrete -n_bins 17 -ded -frames 500000 -info ded_discrete_17bins


# .venv/bin/python toy_domain/run.py -action_mode continuous -agent iqn -ded -info continuous_iqn_2 -alpha 0.2 -frames 50000 -anchor_ratio 0.7 -n_step 200

# LifeGate discrete
# .venv/bin/python toy_domain/run.py -env LifeGate -action_mode discrete -agent iqn -ded -frames 10000 -info lifegate_iqn

# GridNav continuous
# .venv/bin/python toy_domain/run.py -env GridNav -action_mode continuous -agent iqn -frames 100000 -info gridnav_iqn -ded -dead_end_pct 0.01



.venv/bin/python toy_domain/run.py -env MedGrid -action_mode continuous -agent iqn -ded -frames 10 -info medgrid_test_cont medgrid_scale 0.5


# =======================================================================
# MIMIC-IV Sepsis — Continuous IQN (D-network + R-network)
# =======================================================================
# Actions: 2-D continuous [fluid_dose_norm, vasopressor_dose_norm] ∈ [0,1]²
# State  : NCDE hidden-state encoding of patient's clinical time-series
# Output : D-network (dead-end classifier) + R-network (recovery classifier)

# -- Data pipeline (run once before any training) -----------------------
# Step 1: impute heights, create stratified train/val/test splits
# jupyter nbconvert --to notebook --execute notebooks/impute_ht_and_split_data.ipynb

# Step 2: rectilinear interpolation + z-score normalisation
# .venv/bin/python preprocess_ncde_data.py

# Step 3: train NCDE state encoder
# .venv/bin/python train_ncde.py

# Step 4: encode all trajectories into hidden states
# .venv/bin/python encode_data.py

# -- Smoke test: verifies the full training pipeline in ~30 seconds -----
# Runs 2 epochs on 500 transitions; check for shape errors / NaN losses
.venv/bin/python train_rl.py -c iqn_continuous_mimic -o smoke_test True

# -- Full training: D-network then R-network, 100 epochs each -----------
# .venv/bin/python train_rl.py -c iqn_continuous_mimic

# -- Full training with CQL conservative penalty ------------------------
# .venv/bin/python train_rl.py -c iqn_continuous_mimic -o use_cql True -o cql_weight 0.5
