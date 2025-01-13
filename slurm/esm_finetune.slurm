#!/bin/bash
#SBATCH --job-name=esm_finetune   # Name of the job
#SBATCH --output=esm_finetune_%j.out  # Custom output log file (%j is the job ID)
#SBATCH --error=esm_finetune_%j.err   # Custom error log file
#SBATCH --partition=gpu             # Partition to submit to (e.g., gpu)
#SBATCH --gres=gpu:1                # Number of GPUs to request (adjust as needed)
#SBATCH --cpus-per-task=4           # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=128G                   # Memory per node (adjust as needed)
#SBATCH --time=04:00:00             # Time limit (hh:mm:ss)
#SBATCH --mail-type=ALL             # Notifications for job events (BEGIN, END, FAIL, etc.)
#SBATCH --mail-user=thiemea@stanford.edu  # Email for notifications

PROJECT_DIR=$HOME/mydata/projects/TabulaSapiens

# load cuda
module load cuda/12.4

# Activate your environment if needed
source activate sc_env

# Run your Python script
python $PROJECT_DIR/scripts/esm_finetune.py $PROJECT_DIR/results/motif/motif_sequence_data.csv $PROJECT_DIR/logs/esm_finetune_$SLURM_JOB_ID.log
