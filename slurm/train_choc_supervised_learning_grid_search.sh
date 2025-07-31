#!/bin/bash
#SBATCH --job-name=xgb_choc_mortality
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=a-faisal

# load miniconda 
module load miniconda/py312_25.1.1-2
module load gnu/14.1

# Set up environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=offline
export WANDB_DIR=/home/38/bt310238/wandb

# Navigate to project directory
cd /groups/a-faisal/bt310238/ai_clinician/code/ai_clinician_pediatric_rl

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rl_env

# Verify environment
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Install/update dependencies if needed
# pip install -r requirements.txt

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Add to your script
echo "CPU info:"
lscpu | grep "Model name"
cat /proc/cpuinfo | grep "model name" | head -1

echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Run the training script
python -m src.scripts.supervised.supervised_learning \
    --hospital choc \
    --model_type xgb



echo "End time: $(date)"
echo "Job completed!"