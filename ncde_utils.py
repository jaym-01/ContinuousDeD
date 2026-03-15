"""
Utility and Helper functions for use to facilitate state aggregation using Online Neural CDEs.

The various scripts in use here are largely in place to finish pre-processing the data and follow
from: https://github.com/jambo6/online-neural-cdes/tree/main/get_data, primarily mimic-iv/prepare.py, common.py and transformers.py

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

March 2022 by Taylor Killian; University of Toronto + Vector Institute
============================================================================================================================
Notes:
 - 


"""

############################################
#           IMPORTS and DEFINITIONS
############################################

import os, sys
from pathlib import Path
import numpy as np

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class TransformerMixin:
    """Minimal sklearn TransformerMixin replacement — provides fit_transform only."""
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

# torchdiffeq / torchsde are only needed for torchcde's ODE solver (cdeint).
# We only use the interpolation functions, so mock these two packages before
# torchcde/__init__.py tries to import them.  This prevents the
# torchdiffeq → scipy.integrate → broken-numpy chain on environments where
# scipy and numpy versions are mismatched (e.g. Colab).
# If they are already properly loaded (healthy scipy), the mocks are skipped.
import sys as _sys
if 'torchdiffeq' not in _sys.modules:
    from unittest.mock import MagicMock as _MM
    _sys.modules['torchdiffeq'] = _MM()
    _sys.modules['torchsde']    = _MM()

from torchcde import linear_interpolation_coeffs, natural_cubic_coeffs


# sys.path.append("../")
# import common  # Commented out for now because I'm planning on wrapping common.py into this file...

############################################
#           HELPER FUNCTIONS
############################################

def open_npz(npz, key):
    """Gets npz files and converts to tensor format."""
    data = npz[key]
    if data.dtype == "O":
        data = [torch.tensor(x, dtype=torch.float32) for x in data]
    else:
        try:
            data = torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            raise Exception(
                "Could not convert key={} to a tensor with error {}.".format(key, e)
            )
    return data

def reduce_tensor_samples(tensors, num_samples=100):
    """Reduce number of samples in each tensor, useful for testing."""
    test_tensors = []
    for tensor in tensors:
        test_tensors.append(tensor[:num_samples])
    return test_tensors


def temporal_pipeline(
    temporal_data,
    intensities,
    interpolation_method="linear",
    return_as_numpy=True,
):
    assert len(temporal_data[0].shape) == 2

    # Apply
    temporal_out = Interpolation(method=interpolation_method).fit_transform(
        temporal_data
    )

    # Concatenate the intensities to the interpolated data
    for ii in range(len(temporal_out)):
        # terminal temporal_out row is not doubled with the rectilinear interpolation, take everything but the last row of the intensities
        temporal_out[ii] = torch.concat([temporal_out[ii], intensities[ii][:-1]], axis=-1) 

    # Pad the tensors to match the maximum length of the observed data sequences
    temporal_out = pad_sequence(temporal_out, batch_first=True, padding_value=0.)

    # Numpy
    if return_as_numpy:
        temporal_out = np.stack(temporal_out).astype(np.float32)
        
    return temporal_out

def define_temporal_labels(temporal_data, num_columns):
    """Assumming `temporal_data` has already been processed"""

    # We only want to keep the actual temporally varying data... Need to subsample since the interpolation doubled things up
    # We also only want the observation columns, not the intensities (minus the time and ventilation --first and last-- columns)
    temp = temporal_data[:, ::2, 1:(num_columns-1)]

    # Shift the observed temporal data by one timestep, remove the time and ventilation (first and last) columns
    # Add an empty row to account for the change of shape...
    return np.concatenate((temp[:,1:,:], np.zeros((temp.shape[0],1,num_columns-2))), axis=1)

def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        return nn.Linear(n_inputs, n_outputs)
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers-1):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

#############################################
#   MODEL TRAINING AND EVALUATION F'ns
#############################################

def trainer(
    model: torch.nn.Module,
    train_loader: DataLoader,
    parameters: Dict[str, float],
    dtype: torch.dtype,
    device: torch.device,
    ) -> nn.Module:
    """Set up and run a full NCDE training loop.
    
    ARGS:
        TODO
        
    """

    model.to(dtype=dtype, device=device)
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=parameters.get("lr", 5e-4),  # Online NCDE documentation recommended starting low
        amsgrad=True)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get('lr_step_size', 30)),
        gamma=parameters.get('lr_gamma', 1.0),  # Default is no learning rate decay
    )

    num_epochs = parameters.get('num_training_epochs', 5)

    # Define an experimentation loop using the testing dataloader
    for epoch in range(num_epochs):
        epoch_loss = []
        print("Experiment: TRAINING...  Epoch = ", epoch+1, 'out of', num_epochs, 'epochs', end="... ")
            # Loop through the data using the data loader
        loss_pred = 0
        for ii, (inputs, masks, lengths, targets, __, __) in enumerate(train_loader):

            # print("Batch {}".format(ii),end='')
            static, temporal, actions = inputs
            static = static.to(device)  # 4 dimensional vector (Gender, Age, Height, Weight)
            temporal = temporal.to(device)    # 77 dimensional vector (time + 38 temporally varying measures + 38 measurement frequency indicators)
            actions = actions.to(device)
            masks = masks.to(device)  # Masks for the actually observed covariates
            lengths = lengths.to(device)
            targets = targets.to(device)

            # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
            max_length = int(lengths.max().item())                  

            # Reduce all tensors by the maximum length    
            temporal = temporal[:,:(2*max_length)-1,:]  # Account for the doubling by rectilinear interpolation
            actions = actions[:,:max_length,:]
            targets = targets[:,:max_length,:]
            masks = masks[:, :max_length, :-1]  # Removing the final column corresponding the ventilator status

            # Re-do the inputs tuple and pass to the model to generate a hidden state and form predictions
            inputs = (static, temporal, actions)

            preds, hidden = model(inputs)

            loss = model.calculate_loss(preds, targets, masks)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_pred += loss.detach().cpu().numpy()

        print("Epoch cumulative loss: ", loss_pred)
    
    # Return the optimized model
    return model


def evaluator(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    dtype: torch.dtype,
    device: torch.device,
    ) -> float:
    """Set up and run a full NCDE evaluation loop.
    
    ARGS:
        TODO
        
    """

    model.eval()

    print("Experiment: EVALUATION (on validation set)", end="... ")
            # Loop through the data using the data loader
    loss_pred = 0
    with torch.no_grad():
        for ii, (inputs, masks, lengths, targets, __, __) in enumerate(eval_loader):
            # print("Batch {}".format(ii),end='')
            static, temporal, actions = inputs
            static = static.to(device)  # 4 dimensional vector (Gender, Age, Height, Weight)
            temporal = temporal.to(device)    # 77 dimensional vector (time + 38 temporally varying measures + 38 measurement frequency indicators)
            actions = actions.to(device)
            masks = masks.to(device)  # Masks for the actually observed covariates
            lengths = lengths.to(device)
            targets = targets.to(device)

            # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
            max_length = int(lengths.max().item())                  

            # Reduce all tensors by the maximum length    
            temporal = temporal[:,:(2*max_length)-1,:]  # Account for the doubling by rectilinear interpolation
            actions = actions[:,:max_length,:]
            targets = targets[:,:max_length,:]
            masks = masks[:, :max_length, :-1]  # Removing the final column corresponding the ventilator status

            # Re-do the inputs tuple and pass to the model to generate a hidden state and form predictions
            inputs = (static, temporal, actions)

            preds, hidden = model(inputs)

            loss = model.calculate_loss(preds, targets, masks)

            loss_pred += loss.detach().cpu().numpy()

    print("Cumulative loss: ", loss_pred)

    return loss_pred


#############################################
#     General Interpolation F'n Class
#############################################

class Interpolation(TransformerMixin):
    """ Linear, rectilinear, cubic, hybrid schemes. """

    def __init__(
        self,
        method="linear",
        channel_indices=None,
        initial_nan_to_zero=True,
        return_as_list=True,
    ):
        """
        Args:
            method (str): One of ("linear", "rectilinear", "cubic", "hybrid").
            channel_indices (list): List of channel indices for the hybrid method.
            initial_nan_to_zero (bool): Set True to mark the initial nan values to be zero.
            return_as_list (bool): Set True to return the data as a list up to final time rather than a padded tensor.
        """
        assert method in [
            "linear",
            "rectilinear",
            "cubic",
            "hybrid",
            "linear_forward_fill",
        ], "Got method {} which is not recognised".format(method)
        if method == "hybrid":
            assert (
                channel_indices is not None
            ), "Hybrid requires specification of the hybrid indices."
            raise NotImplementedError

        self.method = method
        self.channel_indices = channel_indices
        self.initial_nan_to_zero = initial_nan_to_zero
        self.return_as_list = return_as_list

        # Linear interpolation function requires the channel index of times
        self._rectilinear = 0 if self.method == "rectilinear" else None

    def __repr__(self):
        return "{} Interpolation".format(self.method.title())

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # Causality
        if self.initial_nan_to_zero:
            for d in data:
                d[:1, :][torch.isnan(d[:1, :])] = 0.0

        # Build the coeffs
        if self.method == "cubic":

            def func(data):
                return natural_cubic_coeffs(data)

        else:

            def func(data):
                return linear_interpolation_coeffs(data, rectilinear=self._rectilinear)

        # Apply
        if isinstance(data, torch.Tensor):
            coeffs = func(data)
        else:
            coeffs = []
            for d in data:
                coeffs.append(func(d))

        return coeffs

############################################
#         DATA LOADING FUNCTIONS
############################################

def load_data(
    data_dir="/ais/bulbasaur/twkillian/AHE_Sepsis_Data/rectilinear_processed/", 
    use_static=True, overlap=False, 
    batch_size=None, one_hot_actions=False, 
    num_actions=None,
    combine_train_val=False,
    shuffle=False
    ):
    """ TODO: Preamble"""

    # Load the data (stored in .npz format)
    npz = np.load(os.path.join(data_dir, "improved-neural-cdes_data{}.npz".format("_overlapData" if overlap else "")), allow_pickle=True)

    # Pull out the data arrays we'll use to define our dataset, convert all to torch.tensor()
    static_data = torch.tensor(npz['static_data']).to(torch.float) if use_static else None
    temporal_data = torch.tensor(npz['temporal_data']).to(torch.float)
    action_data = torch.tensor(npz['action_data']).to(torch.float)
    outcome_data = torch.tensor(npz['outcomes']).to(torch.float)
    length_data = torch.tensor(npz['lengths']).to(torch.float)
    mask_data = torch.tensor(npz['masks']).to(torch.int)
    label_data =  None if overlap else torch.tensor(npz['labels']).to(torch.float)
    
    stayID_data = torch.tensor(npz['stay_id']).to(torch.float64)


    # Convert actions to one_hot representation if desired
    if one_hot_actions:
        action_data = torch.nn.functional.one_hot(action_data, num_classes=num_actions)

    # Get dimension information
    static_dim = static_data[0].shape[-1] if use_static else None
    action_dim = num_actions if one_hot_actions else action_data.shape[-1]
    input_dim = temporal_data[0].shape[-1]
    # Output dimension corresponds to the number of continuous features observed in the next time step
    # (we predict the full feature set although we'll only record the loss for those that were actually present)
    output_dim = 0 if overlap else label_data[0].shape[-1]
    
    if overlap: # if loading the overlap data, we don't split anything, just create the dataset and construct the DataLoader
        dataset = StaticTemporalDataset(static_data, temporal_data, action_data, outcome_data, length_data, mask_data, label_data, stayID_data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader, test_loader = None, None
        dataloaders = (train_loader, val_loader, test_loader)
    else:  # Split the data into train/val/test, construct datasets and the dataloaders

        # Pull out the train/val/test index splits as previously defined (stratified by AHE/Sepsis and Mortality)
        splits = [npz[x] for x in ("train_idxs", "val_idxs", "test_idxs")]

        if combine_train_val: # Combine the training and validation indices
            splits[0] = np.concatenate((splits[0], splits[1]))

        # Split data
        static_data = (
            [static_data[idxs] for idxs in splits] if use_static else [None, None, None]
        )
        temporal_data = [temporal_data[idxs] for idxs in splits]
        action_data = [action_data[idxs] for idxs in splits]
        outcome_data = [outcome_data[idxs] for idxs in splits]
        length_data = [length_data[idxs] for idxs in splits]
        mask_data = [mask_data[idxs] for idxs in splits]
        label_data = [label_data[idxs] for idxs in splits]
        stayID_data = [stayID_data[idxs] for idxs in splits]

        # Define the datasets and dataloaders
        dataloaders = []
        for i, _ in enumerate(splits):  
            dataset = StaticTemporalDataset(static_data[i], temporal_data[i], action_data[i], outcome_data[i], length_data[i], mask_data[i], label_data[i], stayID_data[i])
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    
    return (
        dataloaders,
        input_dim,
        action_dim,
        static_dim,
        output_dim
    )

class StaticTemporalDataset(Dataset):
    """Construct a Pytorch Dataset from the split and pre-processed MIMIC-IV data"""
    
    def __init__(self, 
        static_data=None, temporal_data=None, 
        action_data=None, outcome_data=None, 
        length_data=None, mask_data=None, 
        label_data=None, stayID_data=None
        ):

        self.static_data = static_data
        self.temporal_data = temporal_data
        self.action_data = action_data
        self.outcome_data = outcome_data
        self.length_data = length_data
        self.mask_data = mask_data
        self.label_data = label_data
        self.stayID_data = stayID_data
    
    def __len__(self):
        return len(self.temporal_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Return the data from each component: we create a tuple from the static, temporal and action data to form the inputs to the models
        # This tuple will be processed and concatenated to form the full inputs to the model.
        if self.static_data is None:
            if self.label_data is None:
                return (None, self.temporal_data[item], self.action_data[item]), self.mask_data[item], self.length_data[item], self.outcome_data[item], self.stayID_data[item]
            else:
                return (None, self.temporal_data[item], self.action_data[item]), self.mask_data[item], self.length_data[item], self.label_data[item], self.outcome_data[item], self.stayID_data[item]
        else:
            if self.label_data is None: 
                return (self.static_data[item], self.temporal_data[item], self.action_data[item]), self.mask_data[item], self.length_data[item], self.outcome_data[item], self.stayID_data[item]
            else:
                return (self.static_data[item], self.temporal_data[item], self.action_data[item]), self.mask_data[item], self.length_data[item], self.label_data[item], self.outcome_data[item], self.stayID_data[item]
        


############################################
#         PREPROCESSING FUNCTIONS
############################################

def process_all_interpolations(static_data, temporal_data, actions, outcomes, masks, intensities, method="rectilinear"):
    """Perform the actual interpolation of the data using `method`.

    Args:
        static_data: tensor of the static features of each subject
        temporal_data: list of tensors of the temporally changing data from each subject
        actions: list of observed clinical actions used over time
        outcomes: list of the outcome (repeated for all time points...)
        masks: list of tensors that represent masks for the non-missing features in `temporal_data`
        intensities: list of tensors that represent the measurement frequencies of the features in `temporal_data`
        method: string denoting the interpolation method. 
            Options - ["linear", "rectilinear", "cubic", "linear_forward_fill"]
                
    Returns:
        processed_data: dictionary of original and interpolated data
    """

    # Create pipelines and apply, save into processed data with ('name', data)
    keys = ["static_data", "temporal_data"]
    processed_data = dict.fromkeys(keys)

    # Static data has already been preprocessed... 
    processed_data["static_data"] = static_data

    # Temporal interpolation. In `temporal_pipeline()` the interpolation of `temporal_data`, concatenation
    # with `intensities``, and padding with zeros to the maximum length sequence is performed
    # Store raw sequences as a numpy object array (variable-length sequences can't form a regular array)
    raw_obj = np.empty(len(temporal_data), dtype=object)
    for i, t in enumerate(temporal_data):
        raw_obj[i] = t.numpy() if torch.is_tensor(t) else np.array(t)
    processed_data["temporal_data_raw"] = raw_obj
    processed_data["temporal_data"] = temporal_pipeline(
            temporal_data, intensities, method, return_as_numpy=True
        )

    # Pad `actions` with zeros to the maximum length sequence
    actions = [torch.tensor(t) for t in actions]
    processed_data['action_data'] = pad_sequence(actions, batch_first=True, padding_value=0.)

    # Pad `outcomes` with zeros to the maximum length sequence
    # First, zero out all intermediate time steps (keeping the -1/+1 at only the final timestep)
    temp = []
    for t in outcomes:
        t[:-1] = 0
        temp.append(torch.tensor(t))
    processed_data['outcomes'] = pad_sequence(temp, batch_first=True, padding_value=0)
    
    # Pad `masks` with zeros to the maximum length sequence
    masks = [torch.tensor(t) for t in masks]
    processed_data['masks'] = pad_sequence(masks, batch_first=True, padding_value=0.)

    return processed_data

def process_interpolate_and_save(name, top_folder):
    """Processing function for a given problem with associated set of labels.
    
    Args: TODO
    
    Returns: TODO
    """
    # Handle save locations
    save_folder = os.path.join(top_folder, name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    fname = os.path.join(save_folder, "improved-neural-cdes_data.npz")

    # Reload the data
    npz = np.load(os.path.join(top_folder,"reduced_format.npz"), allow_pickle=True)
    static_data = open_npz(npz, "static_data")
    temporal_data = open_npz(npz, "temporal_data")
    actions = open_npz(npz, "action_data")
    outcomes = open_npz(npz, "outcome_data")
    masks = open_npz(npz, "masks")
    intensities = open_npz(npz, "intensities")
    lengths = open_npz(npz, "lengths")
    stay_ids = open_npz(npz, "stay_id")

    # Build the interpolations and pad the data
    processed_data = process_all_interpolations(static_data, temporal_data, actions, outcomes, masks, intensities, method='rectilinear')

    # Using the processed temporal data, construct labels for next physiological state prediction (will only focus on the continuous values...)
    processed_labels = define_temporal_labels(processed_data['temporal_data'], num_columns=len(npz['temporal_columns']))

    # Put the train/val/test split indices, actions and lengths in the processed data
    processed_data['train_idxs'] = npz['train_idxs']
    processed_data['val_idxs'] = npz['val_idxs']
    processed_data['test_idxs'] = npz['test_idxs']
    processed_data['lengths'] = lengths
    processed_data['stay_id'] = stay_ids

    # Save
    np.savez(
        fname,
        **processed_data,
        labels = processed_labels
    )

    ### NOW DO THE SAME FOR THE OVERLAP DATA

    # Reload the data from the overlapping dataset
    npzO = np.load(os.path.join(top_folder, "reduced_format_overlapCohort.npz"), allow_pickle=True)
    static_dataO = open_npz(npzO, "static_data")
    temporal_dataO = open_npz(npzO, "temporal_data")
    actionsO = open_npz(npzO, "action_data")
    outcomesO = open_npz(npzO, "outcome_data")
    masksO = open_npz(npzO, "masks")
    intensitiesO = open_npz(npzO, "intensities")
    lengthsO = open_npz(npzO, "lengths")
    stay_idsO = open_npz(npzO, "stay_id")

    # Build the interpolations
    processed_dataO = process_all_interpolations(static_dataO, temporal_dataO, actionsO, outcomesO, masksO, intensitiesO, method='rectilinear')

    processed_dataO['lengths'] = lengthsO
    processed_dataO['stay_id'] = stay_idsO

    # Save
    np.savez(
        fname[:-4]+'_overlapData.npz',
        **processed_dataO,
    )