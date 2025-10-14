#!/bin/bash
# slurm/train_choc_rl_bc_grid_search_single_fold.sh

#SBATCH --job-name=choc_rl_basic_q
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# #SBATCH --partition=normal
# #SBATCH --gres=gpu:l40:1     
#SBATCH --partition=GPU
#SBATCH --gres=gpu:h100:1     
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=j.ormond@imperial.ac.uk

# Exit on any error
set -e

FOLD=0

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
echo "Fold: $FOLD"  # This is your fold number!
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
echo "Starting training for fold $FOLD..."

python -m src.scripts.rl.reinforcement_learning \
    --hospital choc \
    --experiment_type reinforcement_learning \
    --model_type basic_q \
    --mode single_fold \
    --fold 0 \
    --experiment_variant standard \
    --search_variant random_search \
    --n_search_iterations 1 \
    --n_epochs 1000 \
    --action_mode continuous_only \
    --start_time $START_TIME 

exit_code=$?

echo "========================================="
echo "End time: $(date)"
if [ $exit_code -eq 0 ]; then
    echo "Fold $FOLD completed successfully!"
else
    echo "Fold $FOLD failed with exit code $exit_code"
    exit $exit_code
fi
echo "========================================="