#!/bin/bash
#SBATCH --job-name=cnmf_dtc_array          # Name of the job
#SBATCH --output=../runs/%x_%A_%a/cnmf_%A_%a.out  # Custom output log file for array tasks
#SBATCH --error=../runs/%x_%A_%a/cnmf_%A_%a.err   # Custom error log file for array tasks
#SBATCH --partition=cpu                   # Partition to submit to
#SBATCH --cpus-per-task=30              # Number of CPU cores per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=72:00:00                   # Time limit (hh:mm:ss)
#SBATCH --array=0-37                       # Array range (adjust to match the number of tasks) (0-37 for all celltypes)
#SBATCH --mail-user=thiemea@stanford.edu  # Email for notifications

# Variables
PROJECT_DIR=$HOME/mydata/projects/TabulaSapiens
SCRIPT_NAME=run_cnmf_dtc_array.py

# Job-specific directory
JOB_FOLDER=$PROJECT_DIR/runs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Create the directory for this task
mkdir -p $JOB_FOLDER

# Copy the Python script into the task directory
cp $PROJECT_DIR/scripts/$SCRIPT_NAME $JOB_FOLDER/

# Copy the SLURM script into the task directory
cp $0 $JOB_FOLDER/${SLURM_JOB_NAME}_array.sh  # $0 is the name of the current SLURM script

# Change to the task-specific directory
cd $JOB_FOLDER

# Activate the environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sc_env_cnmf

# Run the Python script with the task-specific parameter
python $JOB_FOLDER/$SCRIPT_NAME --task $SLURM_ARRAY_TASK_ID --job $SLURM_ARRAY_JOB_ID