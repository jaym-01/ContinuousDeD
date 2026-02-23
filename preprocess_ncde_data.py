"""
Processing and interpolating all data for use in training the Online NCDE models
"""
import os, sys, time
import yaml
import random
import numpy as np


import torch

from ncde_utils import process_interpolate_and_save
    

if __name__ == '__main__':

    top_folder = '/ais/bulbasaur/twkillian/AHE_Sepsis_Data'
    new_folder = 'rectilinear_processed'

    process_interpolate_and_save(new_folder, top_folder)