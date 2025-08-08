#!/bin/bash
# slurm/train_choc_supervised_learning_grid_search_array_job.sh

#SBATCH --job-name=xgb_choc_mortality
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=j.ormond@imperial.ac.uk

# Exit on any error
set -e

# Generate a unique start time for this entire job array
START_TIME=$(date +"%Y%m%d_%H%M%S")
export START_TIME

# load miniconda 
module load miniconda/py312_25.1.1-2
module load gnu/14.1

# Set up environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=offline
export WANDB_DIR=/home/38/bt310238/wandb

# Navigate to project directory
cd /home/38/bt310238/code/ai_clinician_pediatric_rl

# Create logs directory
mkdir -p slurm/logs

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rl_env

# Print job information
echo "========================================="
echo "Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"  # This is your fold number!
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "========================================="


# Run the training script for this specific fold
echo "Starting training for fold $SLURM_ARRAY_TASK_ID..."

python -m src.scripts.supervised.supervised_learning \
    --hospital choc \
    --experiment_type supervised_learning \
    --model_type xgbregressor \
    --mode single_fold \
    --fold $SLURM_ARRAY_TASK_ID \
    --experiment_variant standard \
    --search_variant random_search \
    --n_search_iterations 30 \
    --start_time $START_TIME 

exit_code=$?

echo "========================================="
echo "End time: $(date)"
if [ $exit_code -eq 0 ]; then
    echo "Fold $SLURM_ARRAY_TASK_ID completed successfully!"
else
    echo "Fold $SLURM_ARRAY_TASK_ID failed with exit code $exit_code"
    exit $exit_code
fi
echo "========================================="