#!/bin/bash

# 1. Submit the continuous training job and capture its Job ID
JOB_CONT=$(qsub medgrid_cont.sh)
echo "Submitted Continuous Training: $JOB_CONT"

# 2. Submit the discrete training job and capture its Job ID
JOB_DISC=$(qsub medgrid_disc.sh)
echo "Submitted Discrete Training: $JOB_DISC"

# 3. Submit the plotting job, conditioned on the successful completion of BOTH prior jobs
JOB_PLOT=$(qsub -W depend=afterok:$JOB_CONT:$JOB_DISC plot_metrics.sh)
echo "Submitted Plotting Job (Waiting on $JOB_CONT and $JOB_DISC): $JOB_PLOT"