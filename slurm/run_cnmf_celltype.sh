#!/bin/bash
#SBATCH --job-name=cnmf_celltype                # Name of the job
#SBATCH --output=../runs/%x_%j/cnmf_%j.out   # Custom output log file
#SBATCH --error=../runs/%x_%j/cnmf_%j.err    # Custom error log file
#SBATCH --partition=cpu                # Partition to submit to (e.g., gpu)
#SBATCH --cpus-per-task=100              # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=128G                     # Memory per node (adjust as needed)
#SBATCH --time=72:00:00                # Time limit (hh:mm:ss)
#SBATCH --mail-user=thiemea@stanford.edu  # Email for notifications

# Variables
PROJECT_DIR=$HOME/mydata/projects/TabulaSapiens
SCRIPT_NAME=run_cnmf_celltype.py
# CONFIG_FILE=$PROJECT_DIR/configs/esm_finetune.yaml

# Job-specific directory
JOB_FOLDER=$PROJECT_DIR/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}

# Create the directory for this job
mkdir -p $JOB_FOLDER

# Copy the Python script into the job directory
cp $PROJECT_DIR/scripts/$SCRIPT_NAME $JOB_FOLDER/

# Copy the config file into the job directory
# cp $CONFIG_FILE $JOB_FOLDER/

# Copy the SLURM script into the job directory
cp $0 $JOB_FOLDER/$SLURM_JOB_NAME.sh  # $0 is the name of the current SLURM script

# Change to the job-specific directory
cd $JOB_FOLDER

# Activate the environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sc_env_cnmf

# Run the Python script
python $JOB_FOLDER/$SCRIPT_NAME