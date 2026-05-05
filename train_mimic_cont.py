import os
import sys
import subprocess
import gzip
import shutil
import glob

# =============================================================================
# Step 1: Install dependencies
# =============================================================================
print("Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'], check=True)

# (Imports requiring the installed dependencies are placed after installation)
import pandas as pd
import numpy as np
import torch
import yaml
import random
import matplotlib.pyplot as plt

# =============================================================================
# Step 2: Build raw NPZ files from CSVs
# =============================================================================
STATIC_COLS   = ['o:gender', 'o:admission_age', 'o:height', 'o:weight']
TEMPORAL_COLS = [
    'm:timestep',
    'o:heart_rate', 'o:sbp', 'o:dbp', 'o:mbp', 'o:resp_rate', 'o:temperature',
    'o:glucose', 'o:so2', 'o:po2', 'o:pco2', 'o:fio2', 'o:pao2fio2ratio',
    'o:ph', 'o:baseexcess', 'o:chloride', 'o:calcium', 'o:potassium', 'o:sodium',
    'o:lactate', 'o:hematocrit', 'o:hemoglobin', 'o:platelet', 'o:wbc', 'o:albumin',
    'o:aniongap', 'o:bicarbonate', 'o:pt', 'o:ptt', 'o:gcs',
    'o:spo2', 'o:bun', 'o:creatinine', 'o:inr', 'o:bilirubin', 'o:alt', 'o:ast',
    'o:UO', 'o:ventilation'
]
ACTION_COLS   = ['a:action_fluid', 'a:action_vaso']
OUTCOME_COL   = ['r:reward']

def read_csv_auto(path):
    """Read CSV whether gzip-compressed or plain text."""
    try:
        with gzip.open(path, 'rt') as f:
            return pd.read_csv(f)
    except Exception:
        return pd.read_csv(path)

def csv_to_npz(csv_path, npz_path):
    """Convert a trajectory CSV to the raw NPZ format required by preprocess_ncde_data.py."""
    print(f"Reading {csv_path} ...")
    df = read_csv_auto(csv_path)
    print(f"  {len(df)} rows, {df['m:stay_id'].nunique()} unique stays")

    static_frame = (
        df[['m:stay_id'] + STATIC_COLS]
        .groupby('m:stay_id', sort=True)
        .first()
    )
    unique_ids = static_frame.index.tolist()

    static_data, temporal_data, action_data, outcome_data = [], [], [], []
    df_by_stay = df.set_index('m:stay_id')

    for sid in unique_ids:
        rows = df_by_stay.loc[[sid]].sort_values('m:timestep')
        static_data.append(static_frame.loc[sid].values.astype(np.float32))
        temporal_data.append(rows[TEMPORAL_COLS].values.astype(np.float32))
        action_data.append(rows[ACTION_COLS].values.astype(np.float32))
        outcome_data.append(rows[OUTCOME_COL].values.astype(np.float32))

    print(f"  Saving to {npz_path} ...")
    np.savez(
        npz_path,
        static_data   = np.stack(static_data),
        temporal_data = np.array(temporal_data, dtype=object),
        action_data   = np.array(action_data,   dtype=object),
        outcome_data  = np.array(outcome_data,  dtype=object),
        static_columns  = STATIC_COLS,
        temporal_columns = TEMPORAL_COLS,
        stay_id         = np.array(unique_ids, dtype=np.float64),
    )
    print(f"  Done ({len(unique_ids)} patients).\n")

DATA_DIR = 'data/continuous_mimic'
csv_to_npz(os.path.join(DATA_DIR, 'final_trajs_noFill.csv'), os.path.join(DATA_DIR, 'reduced_format.npz'))
csv_to_npz(os.path.join(DATA_DIR, 'final_trajs_overlapCohort_noFill.csv'), os.path.join(DATA_DIR, 'reduced_format_overlapCohort.npz'))
print("Both NPZ files created successfully.\n")

# =============================================================================
# Step 3: Preprocess — masks, intensities, normalization, interpolation
# =============================================================================
print("Running preprocessing...")
subprocess.run([sys.executable, 'preprocess_ncde_data.py'], check=True)

# =============================================================================
# Step 4: Train NCDE state encoder
# =============================================================================
from ncde_utils import load_data, evaluator
from ncde import NeuralCDE

NCDE_PARAMS = {
    'hidden_dim':        64,
    'hidden_hidden_dim': 64,
    'pred_num_layers':   3,
    'pred_num_units':    128,
    'lr':                5e-5,
    'num_training_epochs': 50,
    'lr_step_size':      20,
    'lr_gamma':          0.5,
}
CHECKPOINT_EVERY = 5

config_dict = yaml.safe_load(open('./configs/ncde_config.yaml', 'r'))
CONF_DATA_DIR = config_dict['data_dir']
OUTPUT_DIR    = config_dict['output_dir']
BATCH_SIZE    = config_dict['batch_size']
SEED          = config_dict.get('seed', 2022)

os.makedirs(OUTPUT_DIR, exist_ok=True)
NCDE_CKPT_PATH = os.path.join(OUTPUT_DIR, 'ncde_checkpoint.pt')
NCDE_BEST_PATH = os.path.join(OUTPUT_DIR, 'best_model.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

torch.manual_seed(SEED)
random.seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed(SEED)

(train_loader, val_loader, _), input_dim, action_dim, static_dim, output_dim = load_data(
    data_dir=CONF_DATA_DIR, batch_size=BATCH_SIZE
)
(combined_loader, _, _), _, _, _, _ = load_data(
    data_dir=CONF_DATA_DIR, batch_size=BATCH_SIZE, combine_train_val=True, shuffle=True
)

model = NeuralCDE(
    input_dim, NCDE_PARAMS['hidden_dim'], output_dim, static_dim, action_dim,
    hidden_hidden_dim=NCDE_PARAMS['hidden_hidden_dim'],
    pred_num_layers=NCDE_PARAMS['pred_num_layers'],
    pred_num_units=NCDE_PARAMS['pred_num_units'],
    return_sequences=True, device=device
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=NCDE_PARAMS['lr'], amsgrad=True)
"""scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=int(NCDE_PARAMS['lr_step_size']),
    gamma=NCDE_PARAMS['lr_gamma']
)"""

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

start_epoch = 0
if os.path.exists(NCDE_CKPT_PATH):
    print(f"Resuming from checkpoint: {NCDE_CKPT_PATH}")
    ckpt = torch.load(NCDE_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
else:
    print("No checkpoint found — training from scratch.")

num_epochs = NCDE_PARAMS['num_training_epochs']
model.train()

num_epochs = NCDE_PARAMS['num_training_epochs']

for epoch in range(start_epoch, num_epochs):
    #Training phase
    model.train()
    train_loss = 0
    print(f"Epoch {epoch+1}/{num_epochs}", end=" ... ")

    #Iterate over train_loader
    for inputs, masks, lengths, targets, _, _ in train_loader: 
        static, temporal, actions = inputs
        static, temporal, actions = static.to(device), temporal.to(device), actions.to(device)
        masks, lengths, targets   = masks.to(device), lengths.to(device), targets.to(device)

        max_length = int(lengths.max().item())
        temporal = temporal[:, :(2*max_length)-1, :]
        actions  = actions[:, :max_length, :]
        targets  = targets[:, :max_length, :]
        masks    = masks[:, :max_length, :-1]

        preds, _ = model((static, temporal, actions))
        loss = model.calculate_loss(preds, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.detach().cpu().item()
        
    #Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)

    #Validation phase
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inputs, masks, lengths, targets, _, _ in val_loader:
            static, temporal, actions = inputs
            static, temporal, actions = static.to(device), temporal.to(device), actions.to(device)
            masks, lengths, targets   = masks.to(device), lengths.to(device), targets.to(device)

            max_length = int(lengths.max().item())
            temporal = temporal[:, :(2*max_length)-1, :]
            actions  = actions[:, :max_length, :]
            targets  = targets[:, :max_length, :]
            masks    = masks[:, :max_length, :-1]

            preds, _ = model((static, temporal, actions))
            loss = model.calculate_loss(preds, targets, masks)
            
            val_loss += loss.detach().cpu().item()

    #Calculate average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)

    print(f"Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    #Step the scheduler once per epoch using the validation loss
    scheduler.step(avg_val_loss)

    # Save Checkpoints
    if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == num_epochs:
        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'hyperparameters': NCDE_PARAMS,
        }, NCDE_CKPT_PATH)

torch.save({'model': model.state_dict(), 'hyperparameters': NCDE_PARAMS}, NCDE_BEST_PATH)
print(f"\nNCDE model saved to: {NCDE_BEST_PATH}")

# =============================================================================
# Step 5: Encode all trajectories into NCDE hidden states
# =============================================================================
print("\nEncoding data...")
subprocess.run([sys.executable, 'encode_data.py'], check=True)

# Sanity check
ENC_DATA_DIR = 'data/continuous_mimic/rectilinear_processed'
for split in ['train', 'validation', 'test']:
    npz = np.load(os.path.join(ENC_DATA_DIR, f'encoded_{split}.npz'), allow_pickle=True)
    a = npz['actions']
    print(f"  {split:12s}: states {npz['states'].shape}, actions {a.shape}, rewards {npz['rewards'].shape}")

# =============================================================================
# Step 6: Train continuous IQN (D-network + R-network)
# =============================================================================
print("\nTraining Continuous IQN...")
iqn_cmd = [
    sys.executable, 'train_rl.py', '-c', 'iqn_continuous_mimic',
    '-o', 'num_q_hidden_units', '128',
    '-o', 'num_q_layers', '2',
    '-o', 'num_iqn_samples_train', '64',
    '-o', 'num_iqn_samples_est', '64',
    '-o', 'K_actions', '128',
    '-o', 'use_cql', 'True',
    '-o', 'cql_weight', '0.1',
    '-o', 'train_batch_size', '256',
    '-o', 'num_epochs', '75',
    '-o', 'lr', '0.000005',
    '-o', 'num_ps', '2',
    '-o', 'num_ns', '4'
]
subprocess.run(iqn_cmd, check=True)

# Verify checkpoionts
ckpt_dir = 'runs/iqn_continuous_mimic'
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
for c in ckpts:
    size_mb = os.path.getsize(c) / 1e6
    print(f"  {c}  ({size_mb:.1f} MB)")

# =============================================================================
# Step 7: Plot training and validation loss curves (Saved to file)
# =============================================================================
print("\nGenerating loss plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, sided_Q in zip(axes, ['negative', 'positive']):
    loss_path = os.path.join(ckpt_dir, f'q_losses_{sided_Q}.npy')
    ckpt_path = os.path.join(ckpt_dir, f'q_parameters{sided_Q}.pt')

    if os.path.exists(loss_path):
        train_loss = np.load(loss_path)
        ax.plot(train_loss, label='Train loss')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        val_loss = ckpt.get('validation_loss', [])
        if val_loss:
            ax.plot(val_loss, label='Val loss', linestyle='--')

    network_name = 'D-network (dead-end)' if sided_Q == 'negative' else 'R-network (recovery)'
    ax.set_title(network_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Continuous IQN Training Curves — MIMIC-IV Sepsis', fontsize=13)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("Saved: training_curves.png")