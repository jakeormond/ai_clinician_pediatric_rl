"""
Functions for evaluate_dynamic_deephit_final_seeds.ipynb.

Runs inference and analysis for the 5 final Dynamic-DeepHit seeds, for both
variants (actions included in the input / actions omitted). Heavy lifting
(health score, entropy, plotting) is reused from predict.py, and inference
windows are built with the training DynamicDeepHitTransformerDataset;
this module handles inference, a corrected bootstrap, cross-seed reward
agreement, the seed-averaged health score and the vaso-onset event study.

Run discovery, configuration and model loading come from the shared analysis
framework in src/rl/analysis, where dynamic_deephit is a built-in family
(runs.py: discovery and data identity; models.py: loader; data.py: prediction
file names and the ensemble reader):

  study  = make_ddh_study({"with_actions": dir_a, "without_actions": dir_b})
  models = load_ddh_models(study)          # {variant: {seed: LoadedModel}}
  m = models["with_actions"][0]
  m.network               the TransformerDeepHitMulti network (eval mode)
  m.config_manager        the run's config, prepared for this machine, experiment_dir = run folder
  m.record.hyperparams    saved hyperparameters (results.json); m.seed from the same place

Every function below that takes `models` accepts one variant's {seed: LoadedModel}
dict (or a list of LoadedModels).

Place in src/supervised_learning/ (next to predict.py).
"""
import os
import gc
import json
from datetime import datetime
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

from ..data.data_processor_for_shards import ShardProcessor
from .training import create_processor_from_shards
from ..rl.replay_buffer import DynamicDeepHitTransformerDataset
from . import predict as ddh_predict
from ..rl.analysis.runs import DDH_FAMILY, Study, make_study
from ..rl.analysis.models import LoadedModel, load_study_models
from ..rl.analysis.data import (  # noqa: F401  (re-exported for the notebook)
    ddh_prediction_path as prediction_path, is_ensemble_file, load_ensemble_health_score,
)
from .dynamic_deephit_evaluation import (
    calculate_c_index, calculate_brier_score, calculate_roc_and_prc, calculate_all_bin_ece,
)

VARIANTS = {"with_actions": True, "without_actions": False}


def _as_bool(x):
    return str(x).strip().lower() == "true"


def make_ddh_study(variant_dirs, name="ddh", strict=True) -> Study:
    """
    Study of DDH runs grouped by variant. variant_dirs: {"with_actions": dir, "without_actions": dir};
    each dir is searched recursively for run folders. Checks (printed): seeds unique, same
    seed set in both variants, hyperparameters identical within a variant, and only
    include_actions_in_state differing between variants. Each group's label must agree
    with its runs' include_actions_in_state.
    """
    study = make_study(variant_dirs, family=DDH_FAMILY, name=name)
    for (variant,), seed, rec in study.items():
        if variant in VARIANTS and _as_bool(rec.hyperparams.get("include_actions_in_state")) != VARIANTS[variant]:
            raise ValueError(f"{rec.label}: in group {variant!r} but include_actions_in_state="
                             f"{rec.hyperparams.get('include_actions_in_state')!r}")
    # actions_shift may also differ between variants: it only controls final-row masking of the
    # action columns, which the training Dataset removes entirely when include_actions_in_state
    # is 'False', so it has no effect on the action-free runs. It is still checked within each variant.
    study.check(expected_between=("_group", "include_actions_in_state", "actions_shift"), strict=strict)
    return study


def load_ddh_models(study: Study, device=None, verbose=True):
    """{variant: {seed: LoadedModel}} for every run in a study from make_ddh_study."""
    return {key[0]: seeds for key, seeds in load_study_models(study, device=device, verbose=verbose).items()}


def _seed_models(models):
    """[(seed, LoadedModel)] sorted by seed, from {seed: LoadedModel} or a list of LoadedModels."""
    items = list(models.items()) if isinstance(models, dict) else [(m.seed, m) for m in models]
    seeds = [s for s, _ in items]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    for s, m in items:
        if m.seed != s:
            raise ValueError(f"model keyed as seed {s} records seed {m.seed} ({m.record.run_dir})")
    return sorted(items, key=lambda kv: kv[0])


def _net(m: LoadedModel):
    if m.network is None:
        raise ValueError(f"{m.record.label} is not a Dynamic-DeepHit model")
    return m.network


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


def check_action_features(m: LoadedModel, data, split="test"):
    """Standalone input check for one LoadedModel, printing what the model actually receives."""
    config_manager = m.config_manager
    dataset = build_inference_dataset(config_manager, data, split)
    found = check_inputs(_net(m), config_manager, dataset)
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

    models: ONE variant's {seed: LoadedModel}.
    data_by_split: {split: loaded data} (e.g. from load_datasets).

    For each split, the training Dataset is built once (its constructor handles
    action removal and masking settings) and shared by all seeds, which must have
    identical input settings. Windows are then built in vectorised batches on the
    model's device and checked against Dataset.__getitem__ on a sample of rows.
    Each seed's predictions are written to <run folder>/<split>_predictions.h5
    and freed. The loaded data is not modified. Existing files are skipped
    unless overwrite=True.
    """
    items = _seed_models(models)
    settings = {_input_settings(m.config_manager) for _, m in items}
    if len(settings) != 1:
        raise ValueError(f"Seeds have different input settings, cannot share a Dataset: {settings}")
    device = next(_net(items[0][1]).parameters()).device

    for split in splits:
        todo = [(s, m) for s, m in items
                if overwrite or not os.path.exists(prediction_path(m.config_manager.experiment_dir, split))]
        if not todo:
            print(f"{split}: all seeds already done, skipping")
            continue
        dataset = build_inference_dataset(todo[0][1].config_manager, data_by_split[split], split)
        trajectories = dataset.trajectories.to(device)
        row_starts = _row_starts(dataset, device)
        print(f"{split}: {dataset.n_windows} windows, {len(dataset.feature_names) + 1} inputs, "
              f"actions_shift={dataset.actions_shift}")
        if n_verify:
            verify_windows(dataset, trajectories, row_starts, n_check=n_verify)

        for seed, m in todo:
            config_manager = m.config_manager
            check_inputs(_net(m), config_manager, dataset)
            logits, latents = _predict_dataset(_net(m), dataset, trajectories, row_starts, batch_size)
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
    for seed, m in _seed_models(models):
        path = prediction_path(m.config_manager.experiment_dir, "test")
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


def load_dataset(model_or_config, split="test"):
    """Pre-normalised DDH dataset for one split (accepts a LoadedModel or a config manager)."""
    cm = model_or_config.config_manager if isinstance(model_or_config, LoadedModel) else model_or_config
    return ShardProcessor.load_dynamic_deephit_dataset_from_h5(dataset_filepath(cm, split))


def load_datasets(models, splits=("train", "val", "test")):
    """
    {split: dataset} loaded once and shared by every model passed (e.g. both variants:
    the action columns are dropped inside the training Dataset, not in these files).
    Raises if the models would read different dataset files.
    """
    flat = []
    for x in (models.values() if isinstance(models, dict) else models):
        flat.extend(x.values() if isinstance(x, dict) else [x])
    out = {}
    for split in splits:
        paths = {dataset_filepath(m.config_manager, split) for m in flat}
        if len(paths) != 1:
            raise ValueError(f"models read different {split} dataset files: {sorted(paths)}")
        out[split] = ShardProcessor.load_dynamic_deephit_dataset_from_h5(paths.pop())
    return out


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
# Agreement across seeds and the seed-averaged (ensemble) health score
# ----------------------------------------------------------------------------
_HS_ONLY_EXCLUDE = ("compressed_latents", "model_outputs", "predictions", "entropy")


def load_health_scores(models, split):
    """Health score of every seed for one split, loading nothing heavy.

    models: ONE variant's {seed: LoadedModel}.
    Returns (hs_by_seed {seed: (N,) array}, meta) where meta is the first seed's
    loaded dict minus the per-model keys (stay_ids, index_ptr, event, ...).
    Raises if stays or row counts differ between seeds.
    """
    hs_by_seed, meta = {}, None
    for seed, m in _seed_models(models):
        path = prediction_path(m.config_manager.experiment_dir, split)
        preds = ShardProcessor.load_dynamic_deephit_predictions_from_h5(
            path, keys_to_exclude=list(_HS_ONLY_EXCLUDE))
        if meta is None:
            meta = {k: v for k, v in preds.items() if k != "health_score"}
        else:
            for key in ("index_ptr", "stay_ids"):
                if not np.array_equal(np.asarray(preds[key]), np.asarray(meta[key])):
                    raise ValueError(f"{key} differs between seeds ({split}, seed {seed})")
        hs_by_seed[seed] = np.asarray(preds["health_score"], dtype=np.float64)
        del preds
    return dict(sorted(hs_by_seed.items())), meta


def _demean_within_stays(x, index_ptr):
    lengths = np.diff(index_ptr)
    stay_means = np.add.reduceat(x, index_ptr[:-1]) / lengths
    return x - np.repeat(stay_means, lengths)


def _mean_offdiag(m):
    m = np.asarray(m)
    return m[~np.eye(len(m), dtype=bool)].mean()


def seed_agreement_summary(hs_by_seed, index_ptr):
    """How similar the seeds' health scores and rewards are.

    Pooled health-score correlation is dominated by differences between
    patients; the within-stay and reward rows measure agreement on the
    hour-to-hour changes the RL agent actually receives.
    Returns a one-column DataFrame of metrics.
    """
    index_ptr = np.asarray(index_ptr)
    seeds = list(hs_by_seed)
    hs = np.stack([hs_by_seed[s] for s in seeds])                    # (S, N)
    rewards = np.stack([within_stay_reward(h, index_ptr) for h in hs])
    hs_within = np.stack([_demean_within_stays(h, index_ptr) for h in hs])

    across_seed_sd = hs.std(axis=0, ddof=1)
    nonzero = np.all(rewards != 0, axis=0)
    signs = np.sign(rewards[:, nonzero])
    all_same_sign = np.mean(np.all(signs == signs[0], axis=0))

    # each seed's reward vs the mean of the OTHER seeds (avoids self-inflation)
    loo = [stats.pearsonr(rewards[i], np.delete(rewards, i, axis=0).mean(axis=0))[0]
           for i in range(len(seeds))]

    out = {
        "n_seeds": len(seeds),
        "n_rows": hs.shape[1],
        "health_score_pearson_mean": _mean_offdiag(np.corrcoef(hs)),
        "health_score_pearson_min": np.corrcoef(hs)[~np.eye(len(seeds), dtype=bool)].min(),
        "health_score_within_stay_pearson_mean": _mean_offdiag(np.corrcoef(hs_within)),
        "reward_pearson_mean": _mean_offdiag(np.corrcoef(rewards)),
        "reward_pearson_min": np.corrcoef(rewards)[~np.eye(len(seeds), dtype=bool)].min(),
        "reward_vs_other_seeds_mean_pearson": float(np.mean(loo)),
        "reward_all_seeds_same_sign": float(all_same_sign),
        "health_score_sd_between_rows": hs.mean(axis=0).std(),
        "health_score_mean_across_seed_sd": across_seed_sd.mean(),
        "health_score_p95_across_seed_sd": np.percentile(across_seed_sd, 95),
        "share_of_variance_from_seed": (hs.var(axis=0, ddof=1).mean()
                                        / hs.reshape(-1).var(ddof=1)),
        "reward_sd_single_seed_mean": rewards.std(axis=1).mean(),
        "reward_sd_ensemble": rewards.mean(axis=0).std(),
    }
    return pd.DataFrame.from_dict(out, orient="index", columns=["value"])


def plot_seed_trajectories(hs_by_seed, meta, dataset=None, ensemble=None, n_stays=6, rng_seed=0, save_path=None):
    """Health score of every seed, plus the seed mean, for a few random stays.

    Outcomes come from `dataset` (e.g. datasets["test"]), since prediction
    files don't store them; its stay_ids must match the predictions. Without a
    dataset, stays are sampled without regard to outcome.

    ensemble: optional dict from load_ensemble_health_score. If given, the saved
    seed-averaged score (with a +/-1 SD band) is drawn instead of a mean computed
    here, after checking it matches the seeds' own scores.
    """
    rng = np.random.default_rng(rng_seed)
    index_ptr = np.asarray(meta["index_ptr"])
    n_total = len(index_ptr) - 1

    events = None
    if dataset is not None:
        ddd = _dataset_dict(dataset)
        if not np.array_equal(np.asarray(ddd["stay_ids"]), np.asarray(meta["stay_ids"])):
            raise ValueError("stay_ids of the dataset and the predictions differ")
        events = np.asarray(ddd["event"])
    elif "event" in meta:
        events = np.asarray(meta["event"])

    if ensemble is not None:
        if not np.array_equal(np.asarray(ensemble["stay_ids"]), np.asarray(meta["stay_ids"])):
            raise ValueError("stay_ids of the ensemble file and the predictions differ")
        ens_seeds = list(ensemble["seeds"])
        if sorted(ens_seeds) != sorted(hs_by_seed):
            raise ValueError(f"Ensemble averaged seeds {ens_seeds}, predictions have {list(hs_by_seed)}")
        recomputed = np.mean([hs_by_seed[s] for s in ens_seeds], axis=0)
        if not np.allclose(recomputed, ensemble["health_score"], atol=1e-3):
            raise ValueError("Saved ensemble health_score does not match the mean of the seeds' scores")

    if events is not None:
        deaths, survivors = np.flatnonzero(events == 1), np.flatnonzero(events == 0)
        k = n_stays // 2
        stays = np.concatenate([rng.choice(deaths, min(k, len(deaths)), replace=False),
                                rng.choice(survivors, min(n_stays - k, len(survivors)), replace=False)])
    else:
        stays = rng.choice(n_total, size=min(n_stays, n_total), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(stays) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for ax, stay in zip(axes.ravel(), stays):
        a, b = index_ptr[stay], index_ptr[stay + 1]
        hours = np.arange(b - a)
        seed_curves = np.stack([hs[a:b] for hs in hs_by_seed.values()])
        for seed, curve in zip(hs_by_seed, seed_curves):
            ax.plot(hours, curve, lw=0.8, alpha=0.6, label=f"seed {seed}")
        if ensemble is not None:
            ens_mean = np.asarray(ensemble["health_score"][a:b])
            ens_sd = np.asarray(ensemble["health_score_sd"][a:b])
            ax.fill_between(hours, ens_mean - ens_sd, ens_mean + ens_sd, color="k", alpha=0.15, lw=0)
            ax.plot(hours, ens_mean, color="k", lw=2, label="ensemble (saved)")
        else:
            ax.plot(hours, seed_curves.mean(axis=0), color="k", lw=2, label="mean")
        outcome = "" if events is None else (" (died)" if events[stay] == 1 else " (survived)")
        ax.set_title(f"stay {meta['stay_ids'][stay]}{outcome}")
        ax.set_xlabel("Hour of stay")
        ax.set_ylabel("Health score")
    for ax in axes.ravel()[len(stays):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def ensemble_dir(models):
    """The job folder holding all seeds' run folders (and config_manager.pkl).

    Raises if the seeds do not all sit in the same parent folder.
    """
    parents = {os.path.dirname(os.path.normpath(m.record.run_dir)) for _, m in _seed_models(models)}
    if len(parents) != 1:
        raise ValueError(f"Seeds are not in a single job folder: {sorted(parents)}")
    return parents.pop()


def build_ensemble_health_score(models, split, data, out_dir=None, overwrite=False):
    """Average the seeds' health scores for one split and save for RL training.

    data: the loaded dataset for this split (e.g. datasets[split]); supplies
    feature_names and event, which prediction files do not return.

    Written with h5py to <out_dir>/<split>_predictions.h5 using the same dataset
    names/dtypes as ShardProcessor.dynamic_deephit_dataset_to_h5 for the shared
    keys (feature_names, stay_ids, index_ptr, carry_forward_ind, response_zero_ind),
    plus event and:
      health_score        mean across seeds  -> the shared reward source
      health_score_sd     across-seed SD
      health_score_seeds  (N, n_seeds) per-seed scores, columns in the order of the 'seeds' attribute
    No latents are included: RL states still come from each seed's own file.
    The format is documented in src/rl/analysis/data.py, whose load_ensemble_health_score
    (used by analysis and RL training) reads the file back as a check.
    ensemble_info.json records which runs were averaged.

    out_dir defaults to the job folder that holds the seeds' run folders and
    config_manager.pkl (see ensemble_dir).
    """
    if out_dir is None:
        out_dir = ensemble_dir(models)
    os.makedirs(out_dir, exist_ok=True)
    path = prediction_path(out_dir, split)
    if os.path.exists(path) and not overwrite:
        if is_ensemble_file(path):
            print(f"{split}: ensemble exists, skipping ({path})")
            return path
        print(f"{split}: existing file is incomplete, rebuilding ({path})")

    first = _seed_models(models)[0][1]
    hs_by_seed, meta = load_health_scores(models, split)
    ddd = _dataset_dict(data)
    for key in ("stay_ids", "index_ptr"):
        if not np.array_equal(np.asarray(ddd[key]), np.asarray(meta[key])):
            raise ValueError(f"{key} of the loaded {split} dataset and the predictions differ")

    seeds = list(hs_by_seed)
    stacked = np.stack([hs_by_seed[s] for s in seeds], axis=1)        # (N, S)
    mean_hs = stacked.mean(axis=1).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("feature_names",
                         data=np.array([n.encode("utf-8") for n in ddd["feature_names"]], dtype="S"))
        f.create_dataset("stay_ids", data=np.asarray(meta["stay_ids"]), dtype=np.int32)
        f.create_dataset("index_ptr", data=np.asarray(meta["index_ptr"]), dtype=np.int32)
        for key in ("carry_forward_ind", "response_zero_ind"):
            if key in meta:
                f.create_dataset(key, data=np.asarray(meta[key]))
        f.create_dataset("event", data=np.asarray(ddd["event"]))
        f.create_dataset("health_score", data=mean_hs)
        f.create_dataset("health_score_sd", data=stacked.std(axis=1, ddof=1).astype(np.float32))
        f.create_dataset("health_score_seeds", data=stacked.astype(np.float32))
        f.attrs["seeds"] = np.asarray(seeds, dtype=np.int32)
        f.attrs["include_actions_in_state"] = str(first.config_manager.include_actions_in_state)

    # round trip through the shared reader used by analysis and RL training
    back = load_ensemble_health_score(path, keys=("health_score", "stay_ids", "index_ptr"))
    if not (np.allclose(back["health_score"], mean_hs) and back["seeds"] == seeds
            and np.array_equal(back["index_ptr"], np.asarray(meta["index_ptr"]))):
        raise RuntimeError(f"{path}: file read back does not match what was written")

    info_path = os.path.join(out_dir, "ensemble_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({
        "seeds": seeds,
        "run_dirs": {str(seed): m.record.run_dir for seed, m in _seed_models(models)},
        "include_actions_in_state": str(first.config_manager.include_actions_in_state),
        "splits_written": sorted(set(info.get("splits_written", [])) | {split}),
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"{split}: ensemble of seeds {seeds} saved to {path}")
    return path


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