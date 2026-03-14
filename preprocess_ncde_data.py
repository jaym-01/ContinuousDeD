"""
Processing and interpolating all data for use in training the Online NCDE models.

Computes masks, intensities, lengths, and train/val/test splits from the raw NPZ files
(produced by make_npz_files.py), augments them, then runs the rectilinear interpolation
pipeline defined in ncde_utils.process_interpolate_and_save.

Expected input files (in data/continuous_mimic/):
  - reduced_format.npz              (main cohort)
  - reduced_format_overlapCohort.npz (overlap / RL cohort)

Output (in data/continuous_mimic/rectilinear_processed/):
  - improved-neural-cdes_data.npz
  - improved-neural-cdes_data_overlapData.npz
"""

import os
import numpy as np
import pandas as pd
import torch

from ncde_utils import process_interpolate_and_save, open_npz

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT_DIR, 'data', 'continuous_mimic')


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def fill_nans(temporal_list):
    """Forward-fill then backward-fill NaN in each patient's temporal sequence.
    Any remaining NaN (all-missing columns) are set to 0.

    Args:
        temporal_list: list of (T, D) torch tensors or numpy arrays (may contain NaN)
    Returns:
        list of (T, D) torch.float32 tensors with no NaN
    """
    filled = []
    for t in temporal_list:
        arr = t.numpy() if torch.is_tensor(t) else np.array(t, dtype=np.float32)
        df = pd.DataFrame(arr).ffill().bfill().fillna(0.0)
        filled.append(torch.tensor(df.values.astype(np.float32)))
    return filled


def compute_masks_intensities_lengths(raw_temporal_list):
    """Derive masks, intensities, and sequence lengths from raw (NaN-containing) temporal data.

    Args:
        raw_temporal_list: list of (T, D) tensors/arrays where column 0 = time,
                           columns 1..D-1 = clinical features (may contain NaN)

    Returns:
        masks:      list of (T, D-1) float32 numpy arrays  (1=observed, 0=missing)
        intensities: list of (2T, D-1) float32 torch tensors
                     Interleaved for rectilinear interpolation — after rectilinear
                     expands T → 2T-1 rows, we concatenate intensities[:-1] = 2T-1 rows
                     along the feature axis, yielding (2T-1, D-1 + D-1) appended channels.
        lengths:    (N,) float32 numpy array of per-patient sequence lengths
    """
    masks, intensities, lengths = [], [], []
    for t in raw_temporal_list:
        arr = t.numpy() if torch.is_tensor(t) else np.array(t, dtype=np.float32)
        T = arr.shape[0]

        # Observation mask: exclude column 0 (time), shape (T, D-1)
        obs_bool = ~np.isnan(arr[:, 1:])
        obs = obs_bool.astype(np.float32)
        masks.append(obs)

        # Intensity = cumulative observation count per feature (matches original notebook).
        # Interleave for rectilinear: row i → row 2i and 2i+1, giving (2T, D-1).
        # temporal_pipeline slices [:-1] → (2T-1, D-1), matching the 2T-1 rectilinear output.
        cumsum = obs_bool.cumsum(axis=0).astype(np.float32)  # (T, D-1)
        interleaved = np.repeat(cumsum, 2, axis=0)            # (2T, D-1)
        intensities.append(torch.tensor(interleaved))

        lengths.append(float(T))

    return masks, intensities, np.array(lengths, dtype=np.float32)


def normalize_features(temporal_list, static_arr, train_idxs):
    """Z-score normalize temporal features and static features using training-set statistics.

    Temporal column 0 (time) and the last column (ventilation, binary) are left unchanged.
    All continuous columns in between (1..-1) are normalized.
    Static features (gender, age, height, weight) are all normalized together.

    Returns:
        norm_temporal: list of (T, D) float32 tensors, features in [-clip, +clip] range
        norm_static:   (N, 4) float32 array, normalized
        stats:         dict with 'temporal_mean', 'temporal_std', 'static_mean', 'static_std'
    """
    CLIP = 5.0  # clip at ±5 std to prevent extreme values reaching the ODE solver

    # ── Temporal ──────────────────────────────────────────────────────────────
    # Concatenate all training sequences to compute per-feature statistics
    train_arrs = [temporal_list[i].numpy() if torch.is_tensor(temporal_list[i])
                  else np.array(temporal_list[i], dtype=np.float32)
                  for i in train_idxs]
    train_concat = np.concatenate(train_arrs, axis=0)  # (total_train_steps, D)

    feature_slice = slice(1, -1)   # columns 1..D-2 (skip time col 0 and binary ventilation col -1)
    t_mean = np.nanmean(train_concat[:, feature_slice], axis=0)  # (D-2,)
    t_std  = np.nanstd( train_concat[:, feature_slice], axis=0)
    t_std[t_std < 1e-6] = 1.0

    norm_temporal = []
    for t in temporal_list:
        arr = t.numpy() if torch.is_tensor(t) else np.array(t, dtype=np.float32)
        out = arr.copy()
        out[:, feature_slice] = (arr[:, feature_slice] - t_mean) / t_std
        out[:, feature_slice] = np.clip(out[:, feature_slice], -CLIP, CLIP)
        norm_temporal.append(torch.tensor(out, dtype=torch.float32))

    # ── Static ────────────────────────────────────────────────────────────────
    s_train = static_arr[train_idxs]
    s_mean  = np.nanmean(s_train, axis=0)   # nanmean handles all-NaN columns → NaN
    s_std   = np.nanstd( s_train, axis=0)
    s_std[s_std < 1e-6] = 1.0
    s_mean  = np.where(np.isnan(s_mean), 0.0, s_mean)  # all-NaN column → center on 0
    norm_static = (static_arr - s_mean) / s_std
    norm_static = np.where(np.isnan(norm_static), 0.0, norm_static).astype(np.float32)

    stats = dict(temporal_mean=t_mean, temporal_std=t_std,
                 static_mean=s_mean,   static_std=s_std)
    return norm_temporal, norm_static, stats


def stratified_split(outcome_list, train_frac=0.70, val_frac=0.15, seed=42):
    """70 / 15 / 15 train / val / test split stratified by terminal outcome (mortality).

    Args:
        outcome_list: list of (T, 1) numpy arrays; final reward < 0 → died, ≥ 0 → survived
    Returns:
        train_idxs, val_idxs, test_idxs: sorted integer index arrays
    """
    rng    = np.random.RandomState(seed)
    labels = np.array([float(o[-1, 0]) for o in outcome_list])
    died   = np.where(labels < 0)[0]
    surv   = np.where(labels >= 0)[0]

    def _split(idx):
        idx   = rng.permutation(idx)
        n_tr  = int(len(idx) * train_frac)
        n_va  = int(len(idx) * val_frac)
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    tr1, va1, te1 = _split(died)
    tr2, va2, te2 = _split(surv)

    return (np.sort(np.concatenate([tr1, tr2])),
            np.sort(np.concatenate([va1, va2])),
            np.sort(np.concatenate([te1, te2])))


def _to_obj_array(list_of_arrays):
    """Pack a list of (possibly variable-length) numpy/torch arrays into a numpy object array."""
    out = np.empty(len(list_of_arrays), dtype=object)
    for i, a in enumerate(list_of_arrays):
        out[i] = a.numpy() if torch.is_tensor(a) else np.array(a, dtype=np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main augmentation routine
# ─────────────────────────────────────────────────────────────────────────────

def augment_npz(npz_path, split=True, norm_stats=None):
    """Load a raw NPZ, compute and inject masks/intensities/lengths (+ optional splits), re-save.

    Args:
        npz_path:    path to the .npz file to augment (modified in-place)
        split:       if True, also compute and store train_idxs/val_idxs/test_idxs
        norm_stats:  if provided, use these normalization statistics instead of computing new ones
                     (used for the overlap cohort so it shares the main-cohort's stats)
    Returns:
        norm_stats dict (only meaningful when split=True)
    """
    print(f"  Loading {npz_path} ...")
    with np.load(npz_path, allow_pickle=True) as npz:
        raw_temporal = open_npz(npz, "temporal_data")   # list of (T, D) tensors
        raw_outcome  = open_npz(npz, "outcome_data")    # list of (T, 1) tensors
        static_arr   = npz['static_data'].copy()         # (N, 4) float32
        saved = {k: npz[k] for k in npz.files}

    print("  Computing masks, intensities, lengths ...")
    # Masks come from the original NaN pattern — before any filling or normalization
    masks, intensities, lengths = compute_masks_intensities_lengths(raw_temporal)

    print("  Forward-filling NaN values in temporal data ...")
    filled_temporal = fill_nans(raw_temporal)

    if split:
        print("  Computing stratified train / val / test splits ...")
        raw_out_np = [o.numpy() for o in raw_outcome]
        tr, va, te = stratified_split(raw_out_np)
        print(f"    Train={len(tr)}, Val={len(va)}, Test={len(te)}")
        saved['train_idxs'] = tr
        saved['val_idxs']   = va
        saved['test_idxs']  = te
        train_idxs = tr
    else:
        # For overlap cohort use all indices as "training" for norm stat lookup
        train_idxs = np.arange(len(filled_temporal))

    print("  Normalizing features ...")
    if norm_stats is None:
        norm_temporal, norm_static, norm_stats = normalize_features(
            filled_temporal, static_arr, train_idxs)
    else:
        # Apply pre-computed stats (from main cohort)
        t_mean = norm_stats['temporal_mean']
        t_std  = norm_stats['temporal_std']
        s_mean = norm_stats['static_mean']
        s_std  = norm_stats['static_std']
        CLIP   = 5.0
        norm_temporal = []
        for t in filled_temporal:
            arr = t.numpy() if torch.is_tensor(t) else np.array(t, dtype=np.float32)
            out = arr.copy()
            out[:, 1:-1] = np.clip((arr[:, 1:-1] - t_mean) / t_std, -CLIP, CLIP)
            norm_temporal.append(torch.tensor(out, dtype=torch.float32))
        norm_static = ((static_arr - s_mean) / s_std).astype(np.float32)

    # Overwrite / add required fields
    saved['static_data']   = norm_static
    saved['temporal_data'] = _to_obj_array(norm_temporal)
    saved['masks']         = _to_obj_array(masks)
    saved['intensities']   = _to_obj_array(intensities)
    saved['lengths']       = lengths

    print("  Re-saving augmented NPZ ...")
    np.savez(npz_path, **saved)
    print("  Done.\n")
    return norm_stats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    top_folder = DATA_DIR
    new_folder = 'rectilinear_processed'

    # ── 1. Main cohort ────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1/3  Augmenting main cohort NPZ")
    print("=" * 60)
    norm_stats = augment_npz(os.path.join(top_folder, "reduced_format.npz"), split=True)

    # Save normalization stats for use at inference time
    np.savez(os.path.join(top_folder, "norm_stats.npz"), **norm_stats)
    print(f"  Normalization stats saved to {top_folder}/norm_stats.npz\n")

    # ── 2. Overlap (RL) cohort ────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2/3  Augmenting overlap cohort NPZ")
    print("=" * 60)
    # Use the same normalization stats as the main cohort
    augment_npz(os.path.join(top_folder, "reduced_format_overlapCohort.npz"),
                split=False, norm_stats=norm_stats)

    # ── 3. Rectilinear interpolation → processed NPZs ────────────────────────
    print("=" * 60)
    print("STEP 3/3  Running rectilinear interpolation pipeline")
    print("=" * 60)
    process_interpolate_and_save(new_folder, top_folder)

    out_dir = os.path.join(top_folder, new_folder)
    print(f"\nAll done!  Processed data saved to: {out_dir}")
    print("  improved-neural-cdes_data.npz")
    print("  improved-neural-cdes_data_overlapData.npz")
