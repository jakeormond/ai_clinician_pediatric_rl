"""
Functions for evaluate_dynamic_deephit_final_seeds.ipynb.

Runs inference and analysis for the 5 final Dynamic-DeepHit seeds, for both
variants (actions included in the input / actions omitted). Heavy lifting
(health score, entropy, plotting) is reused from predict.py, and inference
windows are built with the training DynamicDeepHitTransformerDataset;
this module handles model loading, seed discovery, a corrected bootstrap,
cross-seed reward agreement and the vaso-onset event study.

Place in src/supervised_learning/ (next to predict.py).
"""
import os
import gc
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

from ..data.data_processor_for_shards import ShardProcessor
from .training import create_processor_from_shards
from ..rl.replay_buffer import DynamicDeepHitTransformerDataset
from . import predict as ddh_predict
from .models.dynamic_deephit_transformer import TransformerDeepHitMulti
from .dynamic_deephit_evaluation import (
    calculate_c_index, calculate_brier_score, calculate_roc_and_prc, calculate_all_bin_ece,
)

VARIANTS = {"with_actions": True, "without_actions": False}


# ----------------------------------------------------------------------------
# Seed discovery (seed read from each run's own files, never from folder order)
# ----------------------------------------------------------------------------
def _find_key(obj, keys):
    """Depth-first search for the first matching key in nested dicts/lists."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and not isinstance(obj[k], (dict, list)):
                return obj[k]
        for v in obj.values():
            found = _find_key(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, keys)
            if found is not None:
                return found
    return None


def load_results_json(run_dir):
    with open(os.path.join(run_dir, "results.json"), "r") as f:
        return json.load(f)


def get_run_seed(model_dir, seed_keys=("seed", "random_seed")):
    """Seed of one run, read from results.json, then the checkpoint, then config_manager.pkl."""
    seed = _find_key(load_results_json(model_dir), seed_keys)
    if seed is None:
        ckpt = torch.load(os.path.join(model_dir, "final_model.pt"),
                          map_location="cpu", weights_only=False)
        seed = _find_key(ckpt, seed_keys) if isinstance(ckpt, dict) else None
    if seed is None:
        try:
            seed = _find_key(vars(_load_config_manager(model_dir)), seed_keys)
        except FileNotFoundError:
            pass
    if seed is None:
        raise ValueError(f"Could not find a seed in results.json, the checkpoint or "
                         f"config_manager.pkl for {model_dir}")
    return int(seed)


def discover_seed_runs(final_model_dir, expected_actions_in_state=None,
                       seed_keys=("seed", "random_seed")):
    """Map seed -> run directory for every run under final_model_dir.

    A run directory is any folder containing final_model.pt. The seed is read
    from results.json (falling back to the checkpoint). If results.json records
    actions_in_state, it is checked against the expected variant.
    """
    run_dirs = sorted(root for root, _, files in os.walk(final_model_dir)
                      if "final_model.pt" in files)
    if not run_dirs:
        raise ValueError(f"No final_model.pt found under {final_model_dir}")

    runs = {}
    for run_dir in run_dirs:
        results = load_results_json(run_dir)
        seed = get_run_seed(run_dir, seed_keys)
        if seed in runs:
            raise ValueError(f"Seed {seed} found twice: {runs[seed]} and {run_dir}")

        recorded = _find_key(results, ("actions_in_state", "include_actions_in_state"))
        if recorded is not None and expected_actions_in_state is not None:
            recorded_bool = str(recorded).lower() == "true"
            if recorded_bool != expected_actions_in_state:
                raise ValueError(f"{run_dir} records actions_in_state={recorded}, "
                                 f"expected {expected_actions_in_state}")
        runs[seed] = run_dir
    return dict(sorted(runs.items()))


def _load_config_manager(model_dir):
    """config_manager.pkl from the run folder, or its parent (job) folder."""
    path = os.path.join(model_dir, "config_manager.pkl")
    if not os.path.isfile(path):
        parent_path = os.path.join(os.path.dirname(model_dir), "config_manager.pkl")
        if os.path.isfile(parent_path):
            path = parent_path
        else:
            raise FileNotFoundError(
                f"No config_manager.pkl found at {path} or in the parent directory ({parent_path})."
            )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dynamic_deephit_model(model_dir, expected_actions_in_state=None, device=None):
    """
    Loads one trained Dynamic-DeepHit transformer (+ its config_manager and
    hyperparameters) from a specific, already-resolved run directory.

    Mirrors the BC load_actor(): config_manager.pkl is found in the run folder
    (or its parent), hyperparameters come from results.json, and architecture
    dimensions are read from the checkpoint's own weight shapes, so no data
    processor is needed to rebuild the model.

    expected_actions_in_state: used only as a SANITY CHECK against the loaded
        config_manager's training_config['actions_in_state'] (warning, not error).

    The returned config_manager has experiment_dir set to model_dir, so it can
    be passed straight to run_predictions and predict.py's plotting functions.

    Returns: (model, config_manager, hyperparams)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperparams = load_results_json(model_dir)["hyperparameters"]

    config_manager = _load_config_manager(model_dir)
    config_manager.experiment_dir = model_dir
    recorded = hyperparams.get(
        "include_actions_in_state",
        getattr(config_manager, "training_config", {}).get("actions_in_state", None),
    )
    if expected_actions_in_state is not None and recorded is not None:
        if isinstance(expected_actions_in_state, str):
            raise TypeError("expected_actions_in_state must be True/False, e.g. eval_ddh.VARIANTS[variant]")
        recorded_bool = str(recorded).strip().lower() == "true"
        if recorded_bool != expected_actions_in_state:
            print(f"Warning: expected actions_in_state={expected_actions_in_state} but "
                  f"the run records {recorded!r} for {model_dir}")

    final_model_path = os.path.join(model_dir, "final_model.pt")
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    state_dict = (checkpoint["model_state_dict"]
                  if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)

    # Architecture read directly from the saved weights (cannot disagree with them).
    required = ["input_projection.weight", "heads.0.4.weight", "causal_mask"]
    missing = [k for k in required if k not in state_dict]
    if missing:
        raise KeyError(f"Checkpoint at {final_model_path} is missing {missing}; "
                       f"available keys include {list(state_dict)[:15]}")
    feature_dim = state_dict["input_projection.weight"].shape[1]   # already includes the +1 local-time feature
    num_time_bins = state_dict["heads.0.4.weight"].shape[0]
    num_targets = len({k.split(".")[1] for k in state_dict if k.startswith("heads.")})
    max_seq_len = state_dict["causal_mask"].shape[0]
    n_covariates = (state_dict["covariate_predictor.3.weight"].shape[0]
                    if "covariate_predictor.3.weight" in state_dict else 0)

    time_bins = getattr(config_manager, "time_bins", None)
    if time_bins is not None and len(time_bins) - 1 != num_time_bins:
        raise ValueError(f"config_manager.time_bins has {len(time_bins) - 1} bins but the "
                         f"checkpoint has {num_time_bins} for {model_dir}")
    window_length = getattr(config_manager, "training_config", {}).get("window_length", max_seq_len)
    if window_length != max_seq_len:
        print(f"Warning: window_length={window_length} in config_manager but max_seq_len="
              f"{max_seq_len} in checkpoint for {model_dir}")

    model = TransformerDeepHitMulti(
        feature_dim=feature_dim,
        d_model=hyperparams["d_model"],
        nhead=hyperparams["nhead"],
        num_encoder_layers=hyperparams["num_encoder_layers"],
        ff_factor=hyperparams["ff_factor"],
        num_time_bins=num_time_bins,
        num_targets=num_targets,
        dropout=hyperparams["dropout"],
        max_seq_len=max_seq_len,
        # Only the NUMBER of covariates affects the architecture; the indices
        # themselves are used by the training loss, not the forward pass.
        predict_covariates=list(range(n_covariates)),
        **{k: hyperparams[k] for k in ("prediction_time_horizon", "cnn_kernel_size", "cnn_dilation")
           if k in hyperparams},
    ).to(device)

    # strict: any mismatch here means the model was rebuilt wrongly
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, config_manager, hyperparams


# ----------------------------------------------------------------------------
# Inference inputs: reuse the training Dataset so preprocessing is identical
# ----------------------------------------------------------------------------
_REQUIRED_KEYS = ("trajectories", "feature_names", "stay_ids", "event", "index_ptr",
                  "time_to_event", "time_to_event_raw", "physiological_feature_ind",
                  "physiological_feature_type", "response_zero_ind", "carry_forward_ind")
_EXCLUDE_ON_SAVE = ["trajectories", "time_to_event_raw", "time_to_event"]


def _dataset_dict(data):
    """Accept either a processor (with .dynamic_deephit_dataset) or the dict itself."""
    ddd = data.dynamic_deephit_dataset if hasattr(data, "dynamic_deephit_dataset") else data
    missing = [k for k in _REQUIRED_KEYS if k not in ddd]
    if missing:
        raise KeyError(f"Loaded data is missing keys the training Dataset needs: {missing}")
    return ddd


def build_inference_dataset(config_manager, data, split):
    """DynamicDeepHitTransformerDataset over already-loaded data, in deterministic order.

    data: the loaded split (from load_dataset, or a processor). It is not
    modified. Reusing the training Dataset guarantees inference windows are
    built exactly as in training: action columns removed when
    include_actions_in_state is 'False', final-row masking governed by
    actions_shift, first-step padding and the local-time feature. The Dataset
    normally samples random window ends each epoch; here every row is used once,
    in order, so item i is the window ending at row i.
    """
    if not hasattr(config_manager, "include_actions_in_state"):
        raise AttributeError("config_manager has no include_actions_in_state attribute; "
                             "the training Dataset needs it to decide whether to drop actions.")
    wrapper = SimpleNamespace(dynamic_deephit_dataset=_dataset_dict(data), split=split)
    dataset = DynamicDeepHitTransformerDataset(wrapper, config_manager)
    dataset.window_end_indices = torch.arange(dataset.n_windows)
    dataset.n_samples = dataset.n_windows
    return dataset


def check_inputs(model, config_manager, dataset, action_names=None):
    """Input width must match the model; action features absent iff include_actions_in_state == 'False'."""
    n_inputs = len(dataset.feature_names) + 1          # + local-time feature
    if n_inputs != model.feature_dim:
        raise ValueError(f"Dataset gives {n_inputs} input features but the model expects {model.feature_dim}")
    actions_in_state = str(config_manager.include_actions_in_state).strip().lower() == "true"
    if action_names is None:
        action_names = config_manager.training_config.get("actions") or []
    found = [f for f in dataset.feature_names if any(f.startswith(a) for a in action_names)]
    if actions_in_state and not found:
        raise ValueError("include_actions_in_state is True but no action features are in the inputs")
    if not actions_in_state and found:
        raise ValueError(f"include_actions_in_state is False but action features remain: {found}")
    return found


def check_action_features(model, config_manager, data, split="test"):
    """Standalone input check, printing what the model actually receives."""
    dataset = build_inference_dataset(config_manager, data, split)
    found = check_inputs(model, config_manager, dataset)
    print(f"include_actions_in_state={config_manager.include_actions_in_state!r}, "
          f"actions_shift={dataset.actions_shift}, inputs={len(dataset.feature_names) + 1}, "
          f"action-derived features: {found}")
    del dataset
    gc.collect()
    return found


def _input_settings(config_manager):
    tc = config_manager.training_config
    return (str(config_manager.include_actions_in_state), tc.get("actions_shift", False),
            tc["window_length"], str(tc.get("actions")), tc.get("shuffle_inputs", False))


# ----------------------------------------------------------------------------
# Predictions (using the already-loaded models and data)
# ----------------------------------------------------------------------------
def prediction_path(model_dir, split):
    return os.path.join(model_dir, f"{split}_predictions.h5")


def _row_starts(dataset, device):
    """For every row, the row index where its stay starts."""
    index_ptr = dataset.index_ptr
    lengths = index_ptr[1:] - index_ptr[:-1]
    return torch.repeat_interleave(index_ptr[:-1], lengths).to(device)


def build_windows(dataset, trajectories, row_starts, ends):
    """Vectorised equivalent of DynamicDeepHitTransformerDataset.__getitem__ for a batch of window ends.

    Padding by repeating a stay's first timestep is the same as clamping each
    window's row indices to the stay start, so a whole batch is one gather.
    Applies the same final-row masking (only when actions_shift is False) and
    the same local-time feature as the Dataset.

    trajectories: dataset.trajectories (actions already dropped if applicable), on the target device.
    row_starts:   from _row_starts, same device. ends: (B,) long tensor of window-end rows.
    Returns (B, window_length, n_features + 1).
    """
    W = dataset.window_length
    device = trajectories.device
    k = torch.arange(W, device=device)
    starts = row_starts[ends]
    idx = torch.maximum(ends[:, None] - (W - 1) + k[None, :], starts[:, None])   # (B, W)
    x = trajectories[idx]                                                       # (B, W, F), a new tensor

    if not dataset.actions_shift:
        zero_ind = list(dataset.response_zero_ind)
        cf_ind = list(dataset.carry_forward_ind)
        if len(zero_ind) > 0:
            x[:, -1, zero_ind] = 0.0
        if len(cf_ind) > 0:
            x[:, -1, cf_ind] = x[:, -2, cf_ind]

    pad = (W - (ends - starts + 1)).clamp(min=0)                                # padded steps per window
    local_time = torch.maximum(k[None, :], pad[:, None]).float() / (W - 1)       # (B, W)
    return torch.cat([x, local_time.unsqueeze(-1)], dim=-1)


def verify_windows(dataset, trajectories, row_starts, n_check=256, rng_seed=0):
    """Check build_windows against the Dataset's own __getitem__ on a sample of rows.

    Samples include each checked stay's first rows (where padding applies) and
    random rows. Raises if any window differs.
    """
    rng = np.random.default_rng(rng_seed)
    index_ptr = dataset.index_ptr.numpy()
    stays = rng.choice(len(index_ptr) - 1, size=min(n_check // 4, len(index_ptr) - 1), replace=False)
    first_rows = np.concatenate([np.arange(index_ptr[i], min(index_ptr[i] + 3, index_ptr[i + 1])) for i in stays])
    random_rows = rng.choice(dataset.n_windows, size=n_check, replace=False)
    rows = np.unique(np.concatenate([first_rows, random_rows]))

    ends = torch.as_tensor(rows, dtype=torch.long, device=trajectories.device)
    fast = build_windows(dataset, trajectories, row_starts, ends).cpu()
    for j, r in enumerate(rows):
        ref = dataset[int(r)]["features"]          # window_end_indices = arange, so item r ends at row r
        if not torch.allclose(fast[j], ref, atol=1e-6):
            raise AssertionError(f"Vectorised window differs from Dataset.__getitem__ at row {r}")
    print(f"  verified {len(rows)} windows against Dataset.__getitem__")


@torch.no_grad()
def _predict_dataset(model, dataset, trajectories, row_starts, batch_size):
    """Final-timestep risk logits (N, n_risks, n_bins) and latents (N, d_model) for every window."""
    device = trajectories.device
    model.eval()
    n = dataset.n_windows
    logits_out = np.empty((n, model.num_targets, model.num_time_bins), dtype=np.float32)
    latents_out = np.empty((n, model.d_model), dtype=np.float32)
    for pos in range(0, n, batch_size):
        ends = torch.arange(pos, min(pos + batch_size, n), device=device)
        features = build_windows(dataset, trajectories, row_starts, ends)
        risk_logits, _, output = model(features, padding_mask=None)
        logits = torch.stack([l[:, -1, :] for l in risk_logits], dim=1)
        if torch.isnan(logits).any() or torch.isnan(output).any():
            raise ValueError(f"NaNs in model outputs for rows {pos}-{pos + len(ends)}")
        logits_out[pos:pos + len(ends)] = logits.float().cpu().numpy()
        latents_out[pos:pos + len(ends)] = output.float().cpu().numpy()
    return logits_out, latents_out


def _save_predictions(data, path, logits, latents):
    """Save in the same h5 format as predict.inference_loop, without touching the loaded data."""
    ddd = _dataset_dict(data)
    to_save = {k: v for k, v in ddd.items() if k not in _EXCLUDE_ON_SAVE}   # shallow copy
    logits_t = torch.from_numpy(logits)
    to_save["predictions"] = logits
    to_save["entropy"] = ddh_predict.calculate_prediction_entropy(logits_t).cpu().numpy()
    to_save["health_score"] = ddh_predict.calculate_comprehensive_health_score(logits_t).cpu().numpy()
    to_save["model_outputs"] = latents
    ShardProcessor.dynamic_deephit_dataset_to_h5(
        SimpleNamespace(dynamic_deephit_dataset=to_save), path, keys_to_exclude=_EXCLUDE_ON_SAVE)


def run_predictions(models, data_by_split, splits=("train", "val", "test"),
                    batch_size=1024, overwrite=False, n_verify=256):
    """Predict every split for every loaded model, using data that is already in memory.

    models: list of (model, config_manager, hyperparams, seed) for ONE variant.
    data_by_split: {split: loaded data} (e.g. the `processors` dict from load_dataset).

    For each split, the training Dataset is built once (its constructor handles
    action removal and masking settings) and shared by all seeds, which must have
    identical input settings. Windows are then built in vectorised batches on the
    model's device and checked against Dataset.__getitem__ on a sample of rows.
    Each seed's predictions are written to <run folder>/<split>_predictions.h5
    and freed. The loaded data is not modified. Existing files are skipped
    unless overwrite=True.
    """
    settings = {_input_settings(cm) for _, cm, _, _ in models}
    if len(settings) != 1:
        raise ValueError(f"Seeds have different input settings, cannot share a Dataset: {settings}")
    device = next(models[0][0].parameters()).device

    for split in splits:
        todo = [m for m in models
                if overwrite or not os.path.exists(prediction_path(m[1].experiment_dir, split))]
        if not todo:
            print(f"{split}: all seeds already done, skipping")
            continue
        dataset = build_inference_dataset(todo[0][1], data_by_split[split], split)
        trajectories = dataset.trajectories.to(device)
        row_starts = _row_starts(dataset, device)
        print(f"{split}: {dataset.n_windows} windows, {len(dataset.feature_names) + 1} inputs, "
              f"actions_shift={dataset.actions_shift}")
        if n_verify:
            verify_windows(dataset, trajectories, row_starts, n_check=n_verify)

        for model, config_manager, _, seed in todo:
            check_inputs(model, config_manager, dataset)
            logits, latents = _predict_dataset(model, dataset, trajectories, row_starts, batch_size)
            path = prediction_path(config_manager.experiment_dir, split)
            _save_predictions(data_by_split[split], path, logits, latents)
            print(f"  seed {seed}: saved to {path}")
            del logits, latents
            gc.collect()

        del dataset, trajectories, row_starts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_test_predictions(models, keys_to_exclude=("compressed_latents", "model_outputs")):
    """Load each seed's saved test predictions: {seed: predictions_dict}.

    Latents ('model_outputs') are excluded by default since the analyses only
    need the risk logits and health score; pass keys_to_exclude=("compressed_latents",)
    to keep them (e.g. for latent probes).
    """
    out = {}
    for _, config_manager, _, seed in models:
        path = prediction_path(config_manager.experiment_dir, "test")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No test predictions for seed {seed}: {path} (run run_predictions first)")
        out[seed] = ShardProcessor.load_dynamic_deephit_predictions_from_h5(
            path, keys_to_exclude=list(keys_to_exclude))
    return out


def dataset_filepath(config_manager, split="test"):
    """Mirrors the dataset naming in predict.plot_random_predictions()."""
    cm = config_manager
    if cm.cohort != "all" and cm.unit != "all":
        name = f"{split}_{cm.cohort}_{cm.unit.lower()}_dynamic_deephit_dataset.h5"
    elif cm.cohort == "all" and cm.unit != "all":
        name = f"{split}_{cm.unit.lower()}_dynamic_deephit_dataset.h5"
    else:
        name = f"{split}_dynamic_deephit_dataset.h5"
    return os.path.join(cm.get_data_dir, cm.data_config["shard_dir"],
                        f"split_{cm.split_num}", "training_set", name)


def load_dataset(config_manager, split="test"):
    return ShardProcessor.load_dynamic_deephit_dataset_from_h5(dataset_filepath(config_manager, split))


def check_alignment(predictions, dataset):
    if not np.array_equal(predictions["index_ptr"], dataset["index_ptr"]):
        raise ValueError("index_ptr of predictions and dataset differ.")
    if predictions["predictions"].shape[0] != dataset["trajectories"].shape[0]:
        raise ValueError("Row counts of predictions and dataset differ.")


# ----------------------------------------------------------------------------
# Bootstrap (corrected version of predict.bootstrap_evaluation)
# ----------------------------------------------------------------------------
def _to_py(results):
    return {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1
            else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in results.items()}


def bootstrap_evaluation_all_samples(predictions, dataset, n_bootstrap_samples=100,
                                     resample_unit="stay", c_index_max_rows=10000,
                                     rng_seed=0, save_dir=None, overwrite=False):
    """Bootstrap that keeps every sample.

    Differences from predict.bootstrap_evaluation:
      - all samples are returned/saved (the original saves only the last one);
      - the C-index is computed on the bootstrap sample (the original drew its
        C-index rows from the full set, independent of the bootstrap sample);
      - resample_unit="stay" resamples whole stays, since hourly rows within a
        stay are strongly correlated; "timestep" reproduces the original scheme;
      - time_to_event/event come from the dataset file, because they are
        excluded when predictions are saved to h5.
    """
    names = ["c_index", "brier_score", "roc_prc", "ece"]
    paths = {n: os.path.join(save_dir, f"{n}_bootstrap_samples_{resample_unit}.json")
             for n in names} if save_dir else {}
    if paths and not overwrite and all(os.path.exists(p) for p in paths.values()):
        return {n: json.load(open(paths[n])) for n in names}

    check_alignment(predictions, dataset)
    logits = torch.from_numpy(np.asarray(predictions["predictions"]))
    index_ptr = np.asarray(dataset["index_ptr"])
    lengths = np.diff(index_ptr)
    time_to_event = torch.from_numpy(np.asarray(dataset["time_to_event"]))
    event_label = torch.from_numpy(np.repeat(np.asarray(dataset["event"]), lengths).astype(np.int32))
    stay_rows = [np.arange(index_ptr[i], index_ptr[i + 1]) for i in range(len(lengths))]

    rng = np.random.default_rng(rng_seed)
    samples = {n: [] for n in names}
    for _ in range(n_bootstrap_samples):
        if resample_unit == "stay":
            stays = rng.choice(len(stay_rows), size=len(stay_rows), replace=True)
            rows = np.concatenate([stay_rows[s] for s in stays])
        elif resample_unit == "timestep":
            rows = rng.choice(logits.shape[0], size=logits.shape[0], replace=True)
        else:
            raise ValueError(resample_unit)
        c_rows = rows if len(rows) <= c_index_max_rows else rng.choice(rows, c_index_max_rows, replace=False)

        samples["c_index"].append(_to_py(calculate_c_index(
            logits[c_rows], time_to_event[c_rows], event_label[c_rows], random_sample=True)))
        samples["brier_score"].append(_to_py(calculate_brier_score(
            logits[rows], time_to_event[rows], event_label[rows])))
        samples["roc_prc"].append(_to_py(calculate_roc_and_prc(
            logits[rows], time_to_event[rows], event_label[rows])))
        samples["ece"].append(_to_py(calculate_all_bin_ece(
            logits[rows].numpy(), time_to_event[rows].numpy(), event_label[rows].numpy())))

    samples = {n: ddh_predict._to_jsonable(v) for n, v in samples.items()}
    for n, p in paths.items():
        with open(p, "w") as f:
            json.dump(samples[n], f, indent=2)
    return samples


def _flatten_scalars(d, prefix=""):
    """Yield (key_path, float) for every scalar in a nested dict; arrays are skipped."""
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten_scalars(v, prefix=f"{key}.")
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            yield key, float(v)
        elif isinstance(v, (list, np.ndarray)) and np.size(v) == 1:
            yield key, float(np.asarray(v).ravel()[0])


def summarise_bootstrap(boot_by_model):
    """boot_by_model: {(variant, seed): samples}. Returns (long_df, summary_df).

    summary_df per variant/metric: mean and SD of the per-seed means (between-seed
    spread) and the 2.5/97.5 percentiles of all pooled samples.
    """
    rows = []
    for (variant, seed), samples in boot_by_model.items():
        for metric, sample_list in samples.items():
            for i, sample in enumerate(sample_list):
                for key, value in _flatten_scalars(sample):
                    rows.append(dict(variant=variant, seed=seed, metric=f"{metric}.{key}",
                                     sample=i, value=value))
    long_df = pd.DataFrame(rows)
    per_seed = long_df.groupby(["variant", "metric", "seed"])["value"].mean().reset_index()
    between = per_seed.groupby(["variant", "metric"])["value"].agg(
        seed_mean="mean", seed_sd="std", n_seeds="count")
    pooled = long_df.groupby(["variant", "metric"])["value"].quantile([0.025, 0.975]).unstack()
    pooled.columns = ["pooled_ci_low", "pooled_ci_high"]
    return long_df, between.join(pooled).reset_index()


# ----------------------------------------------------------------------------
# Health score / reward across seeds
# ----------------------------------------------------------------------------
def within_stay_reward(health_score, index_ptr):
    """r_t = HS_{t+1} - HS_t within stays (boundary differences removed)."""
    diffs = np.diff(health_score)
    valid = np.ones(len(diffs), dtype=bool)
    valid[np.asarray(index_ptr)[1:-1] - 1] = False
    return diffs[valid]


def pairwise_agreement(series_by_model):
    """Pearson and Spearman correlation matrices between models' series."""
    keys = list(series_by_model)
    labels = [f"{k[0]}:{k[1]}" if isinstance(k, tuple) else str(k) for k in keys]
    n = len(keys)
    pearson, spearman = np.eye(n), np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = series_by_model[keys[i]], series_by_model[keys[j]]
            pearson[i, j] = pearson[j, i] = stats.pearsonr(a, b)[0]
            spearman[i, j] = spearman[j, i] = stats.spearmanr(a, b)[0]
    return (pd.DataFrame(pearson, index=labels, columns=labels),
            pd.DataFrame(spearman, index=labels, columns=labels))


def plot_agreement(corr_df, title, save_path=None):
    fig, ax = plt.subplots(figsize=(0.6 * len(corr_df) + 3, 0.6 * len(corr_df) + 2))
    im = ax.imshow(corr_df.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(corr_df)), corr_df.columns, rotation=90)
    ax.set_yticks(range(len(corr_df)), corr_df.index)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr_df.values[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# Vaso-onset event study
# ----------------------------------------------------------------------------
def find_vaso_onsets(dataset, vaso_feature, min_pre_hours=12, zero_tol=1e-6):
    """Absolute row index of the first nonzero vaso hour per stay.

    Zero dose is taken to be the column minimum of the (normalised) vaso feature;
    check the printed firing fraction is plausible before trusting the onsets.
    Stays whose first vaso hour is < min_pre_hours after admission are excluded.
    """
    names = list(dataset["feature_names"])
    col = np.asarray(dataset["trajectories"])[:, names.index(vaso_feature)]
    zero_value = col.min()
    nonzero = col > zero_value + zero_tol
    print(f"'{vaso_feature}': zero-dose value={zero_value:.4f}, "
          f"nonzero fraction={nonzero.mean():.3%} of rows")

    index_ptr = np.asarray(dataset["index_ptr"])
    onsets, n_with_vaso = [], 0
    for i in range(len(index_ptr) - 1):
        start, end = index_ptr[i], index_ptr[i + 1]
        hits = np.flatnonzero(nonzero[start:end])
        if hits.size:
            n_with_vaso += 1
            if hits[0] >= min_pre_hours:
                onsets.append((i, start + hits[0]))
    print(f"{n_with_vaso} stays received vaso; {len(onsets)} onsets with >= {min_pre_hours}h before")
    return onsets


def event_study_matrix(health_score, index_ptr, onsets, pre=12, post=24):
    """(n_onsets, pre+post+1) health scores aligned on onset; NaN outside the stay."""
    out = np.full((len(onsets), pre + post + 1), np.nan)
    for r, (stay, row) in enumerate(onsets):
        start, end = index_ptr[stay], index_ptr[stay + 1]
        for c, offset in enumerate(range(-pre, post + 1)):
            if start <= row + offset < end:
                out[r, c] = health_score[row + offset]
    return out


def plot_event_study(hs_by_model, index_ptr, onsets, pre=12, post=24, save_path=None):
    """Two panels: HS change from the hour before onset, and mean hourly reward.

    Hour 0 = first vaso hour. Whether an action-aware model sees that dose at
    hour 0 or +1 depends on actions_shift; a step at onset that is absent from
    the action-free model is the input-leak signature.
    """
    hours = np.arange(-pre, post + 1)
    colours = {"with_actions": "tab:red", "without_actions": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for variant in colours:
        seed_curves_hs, seed_curves_r = [], []
        for (v, seed), hs in hs_by_model.items():
            if v != variant:
                continue
            m = event_study_matrix(hs, index_ptr, onsets, pre, post)
            delta = m - m[:, [pre - 1]]
            curve_hs = np.nanmean(delta, axis=0)
            curve_r = np.nanmean(np.diff(m, axis=1), axis=0)
            seed_curves_hs.append(curve_hs)
            seed_curves_r.append(curve_r)
            axes[0].plot(hours, curve_hs, color=colours[variant], alpha=0.25, lw=1)
            axes[1].plot(hours[1:], curve_r, color=colours[variant], alpha=0.25, lw=1)
        if seed_curves_hs:
            axes[0].plot(hours, np.mean(seed_curves_hs, axis=0), color=colours[variant], lw=2.5, label=variant)
            axes[1].plot(hours[1:], np.mean(seed_curves_r, axis=0), color=colours[variant], lw=2.5, label=variant)
    for ax in axes:
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Hours relative to vaso onset")
        ax.legend()
    axes[0].set_ylabel("Health score change from hour -1")
    axes[1].set_ylabel("Mean reward  HS(t) - HS(t-1)")
    axes[0].set_title(f"Health score around vaso onset (n={len(onsets)} onsets; thin = seeds)")
    axes[1].set_title("Hourly reward around vaso onset")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig