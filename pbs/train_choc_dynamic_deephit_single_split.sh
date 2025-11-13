#!/bin/bash
# pbs/train_choc_dynamic_deephit_single_split.sh

#PBS -N choc_dynamic_deephit
#PBS -o pbs/logs/choc_dynamic_deephit.out
#PBS -e pbs/logs/choc_dynamic_deephit.err
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1:gpu_type=h100
#PBS -l walltime=08:00:00
#PBS -m ae
#PBS -M j.ormond@imperial.ac.uk

# Exit on any error
set -e

FOLD=0

# Generate a unique start time for this entire job
START_TIME=$(date +"%Y%m%d_%H%M%S")
export START_TIME

# Load modules (adjust module names as needed for your PBS system)
module load miniconda/py312_25.1.1-2
module load gnu/14.1

# Set up environment variables
export OMP_NUM_THREADS=$NCPUS
export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb

# Navigate to project directory
cd $PBS_O_WORKDIR/../../

# Create logs directory
mkdir -p pbs/logs

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rl_env

# Print job information
echo "========================================="
echo "Job Information:"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "CPUs: $NCPUS"
echo "Memory: 64GB"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "========================================="

# Run the training script for this specific fold
echo "Starting training..."

python -m src.scripts.supervised.supervised_learning \
    --hospital choc \
    --experiment_type supervised_learning \
    --model_type dynamic_deephit \
    --prediction_type discharge_mortality_tte \
    --mode single_split \
    --testing \
    --search_variant fixed_hyperparameters \
    --n_epochs 500 \
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
