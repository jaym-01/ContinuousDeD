#!/bin/bash
#PBS -N plot_metrics
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=1:mem=4gb
# Combine standard output and standard error into one file
#PBS -j oe
# Name the output log file so it's easy to find
#PBS -o plot_metrics.log

# Load the exact same Python module you used to create the environment
module load Python/3.12.3-GCCcore-13.3.0

# Change to the directory where the job was submitted from
cd /home/ContinuousDeD

# Activate your virtual environment
source /home/rl_venv/bin/activate

# Execute your python script
python plot_metrics.py