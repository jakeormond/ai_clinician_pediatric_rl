# inference.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from eval_utils import logits_to_prob_per_timestep, probs_to_survival_and_ci, survival_summaries

@torch.inference_mode()
def run_predictions(model, dataset, batch_size=64, device="cuda", num_workers=4, on_batch_done=None):
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=getattr(dataset, "collate_fn", None))
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        pid = batch["patient_id"]
        t_index = batch["t_index"]

        logits = model(x, mask=mask)
        p = logits_to_prob_per_timestep(logits)
        S, CI = probs_to_survival_and_ci(p, mask)
        med_t, exp_t = survival_summaries(S, mask)

        # assemble rows (omitted for brevity)
        #...
        if on_batch_done:
            on_batch_done(batch, p, S, CI)  # optional callback for streaming/metrics

    df = pd.DataFrame(rows)
    return df