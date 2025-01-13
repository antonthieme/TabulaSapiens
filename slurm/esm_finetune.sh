#!/bin/bash
#SBATCH --job-name=esm_finetune        # Name of the job
#SBATCH --output=../runs/%x_%j/esm_finetune_%j.out   # Custom output log file
#SBATCH --error=../runs/%x_%j/esm_finetune_%j.err    # Custom error log file
#SBATCH --partition=gpu                # Partition to submit to (e.g., gpu)
#SBATCH --gres=gpu:4                   # Number of GPUs to request (adjust as needed)
#SBATCH --cpus-per-task=16              # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=128G                     # Memory per node (adjust as needed)
#SBATCH --time=24:00:00                # Time limit (hh:mm:ss)
#SBATCH --mail-type=ALL                # Notifications for job events (BEGIN, END, FAIL, etc.)
#SBATCH --mail-user=thiemea@stanford.edu  # Email for notifications

# Variables
PROJECT_DIR=$HOME/mydata/projects/TabulaSapiens
SCRIPT_NAME=esm_finetune.py
CONFIG_FILE=$PROJECT_DIR/configs/esm_finetune.yaml

# Job-specific directory
JOB_FOLDER=$PROJECT_DIR/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}

# Create the directory for this job
mkdir -p $JOB_FOLDER

# Copy the Python script into the job directory
cp $PROJECT_DIR/scripts/$SCRIPT_NAME $JOB_FOLDER/

# Copy the config file into the job directory
cp $CONFIG_FILE $JOB_FOLDER/

# Copy the SLURM script into the job directory
cp $0 $JOB_FOLDER/$SLURM_JOB_NAME.sh  # $0 is the name of the current SLURM script

# Change to the job-specific directory
cd $JOB_FOLDER

# Load CUDA module
module load cuda/12.4  # Adjust as needed

# Activate the environment
source activate sc_env

# Run the Python script
python $JOB_FOLDER/$SCRIPT_NAME $CONFIG_FILE $SLURM_JOB_ID
