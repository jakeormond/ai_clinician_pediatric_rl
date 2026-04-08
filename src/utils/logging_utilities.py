import uuid
import wandb
from typing import Dict
import numpy as np
import os

def wandb_init(config: dict, hyperparams: dict) -> None:
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
        id=str(uuid.uuid4())[:8],
        reinit='finish_previous'
    )
    return run
