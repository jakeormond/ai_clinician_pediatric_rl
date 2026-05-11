import uuid
import wandb
from typing import Dict
import numpy as np
import os
import hashlib


def generate_run_seed(job_id: str, search_idx: int) -> int:
    """Generates a deterministic seed based on Job ID and iteration."""
    # Use a hash to turn any string job_id into a stable integer
    if search_idx is None:
        search_idx = 0
    hash_obj = hashlib.sha256(f"{job_id}_{search_idx}".encode())
    return int(hash_obj.hexdigest(), 16) % (2**32)


def get_job_id(config) -> str:
    """Determines the job ID based on the environment."""
    if config.environment == 'local':
        print("Warning: Running in local environment. Using 'local' as job ID.")
        return 'local'
    elif config.environment == 'bayreuth_hpc':
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id is None:
            raise ValueError("SLURM_JOB_ID not found in environment variables.")
        print(f"Running on Bayreuth HPC with Job ID: {job_id}")
        return job_id
    elif config.environment == 'imperial_hpc':
        job_id = os.environ.get('PBS_JOB_ID')
        if job_id is None:
            # raise ValueError("PBS_JOB_ID not found in environment variables.")
            import time
        import socket
        # Create a unique ID from timestamp and hostname
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        host = socket.gethostname().split('.')[0]
        job_id = f"unknown_{host}_{timestamp}"
        print(f"Running on Imperial HPC with Job ID: {job_id}")
        return job_id
    else:
        raise ValueError(f"Unknown environment: {config.environment}")
    


def wandb_init(hyperparams: dict, job_id: str) -> None:
    # 1. Extract metadata for init arguments
    # We use .get() so it doesn't crash if a key is missing
    project = hyperparams.get('project', 'AI_Clinician')
    group = hyperparams.get('group', 'rl_search')
    name = hyperparams['name']
    mode = hyperparams['wandb_mode']

    run = wandb.init(
        project=project,
        group=group,
        name=name,
        mode=mode,
        config=hyperparams,
        id=f"{job_id}_{str(uuid.uuid4())[:8]}",
        reinit='finish_previous'
    )
    return run
