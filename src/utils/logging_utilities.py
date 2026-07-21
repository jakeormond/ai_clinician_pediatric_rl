import uuid
import wandb
from typing import Dict
import numpy as np
import os
import hashlib
import socket
import time


def can_reach_wandb(timeout=3):
    """Check whether the wandb server is reachable (e.g. HPC compute node may lack internet)."""
    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=timeout)
        return True
    except OSError:
        return False


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

    run_id = f"{job_id}_{str(uuid.uuid4())[:8]}"
    init_timeout = int(hyperparams.get("wandb_init_timeout", 180))
    max_retries = int(hyperparams.get("wandb_init_retries", 1))
    retry_sleep_s = float(hyperparams.get("wandb_init_retry_sleep_s", 5.0))

    init_kwargs = {
        "project": project,
        "group": group,
        "name": name,
        "mode": mode,
        "config": hyperparams,
        "id": run_id,
        "reinit": "finish_previous",
        "settings": wandb.Settings(init_timeout=init_timeout),
    }

    run = None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            run = wandb.init(**init_kwargs)
            break
        except Exception as exc:
            last_error = exc
            print(
                f"W&B init failed in mode='{mode}' (attempt {attempt + 1}/{max_retries + 1}): {exc}"
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_s)

    # If online init repeatedly fails (common on busy HPC networks), fall back to offline.
    if run is None and mode == "online":
        print("Falling back to W&B offline mode after online init failure.")
        try:
            init_kwargs["mode"] = "offline"
            run = wandb.init(**init_kwargs)
        except Exception as exc:
            last_error = exc

    # Final safeguard: continue training even if W&B cannot initialize.
    if run is None:
        print(f"W&B disabled for this run due to init failure: {last_error}")
        return None

    # 2. Define custom X-axes directly on the run object
    run.define_metric("global_step")
    run.define_metric("epoch")

    # 3. Bind your metric prefixes to their respective X-axes
    # High-frequency training metrics follow the step-by-step iteration axis
    run.define_metric("train/batch_*", step_metric="iteration")

    # Low-frequency macro trends follow the clean epoch axis
    run.define_metric("train/epoch_*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")

    return run
