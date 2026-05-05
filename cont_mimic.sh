#!/bin/bash
#PBS -N cont_mimic
#PBS -l walltime=09:00:00
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=1
# Combine standard output and standard error into one file
#PBS -j oe
# Name the output log file so it's easy to find
#PBS -o cont_mimic.log

# Load the exact same Python module you used to create the environment
module load Python/3.12.3-GCCcore-13.3.0

# Change to the directory where the job was submitted from
cd /rds/general/user/kd925/home/ContinuousDeD

# Activate your virtual environment
source /rds/general/user/kd925/home/rl_venv/bin/activate

# Execute your python script
python train_mimic_cont.py
python test_compare_mimic_cont.py