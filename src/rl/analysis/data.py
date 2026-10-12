"""
src/rl/analysis/data.py

Processors (the RL datasets models were trained and are evaluated on), built from each
run's DATA IDENTITY (runs.py) and cached, plus canonical per-row helpers and checks.

  store = DataStore()                                   # one per notebook session
  proc  = store.processor(model, split="val")           # the processor for THIS model's data
  paired = store.pair(models, split="val")              # {group: {seed: (model, processor)}}
  train = store.reference("train")                      # one shared train split (see below)

Why by data identity: a model must be evaluated on the inputs it was trained on (its own
Dynamic-DeepHit state embedding and its reward). Runs with the same identity (e.g. every
alpha of a sweep at seed k) share one cached processor.

Reward signature: the reward is computed from hyperparameters (scales, clips, ...). The
cache key therefore also includes the run's reward-related effective hyperparameters, so
runs trained with different reward settings never share a processor by accident.

Per-row helpers (states, actions, rewards, health score aligned to RL rows, trajectory ids,
severity bins) are the single place these are computed; analysis code should use them.

Checks:
  check_rows_aligned       same rows/actions across processors; what differs (states, rewards)
  bellman_check            each critic should be most Bellman-consistent on its own data
"""

from __future__ import annotations

import gc
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .runs import RunRecord

try:                       # optional at import time so the numpy helpers work without torch
    import torch
except ImportError:        # pragma: no cover
    torch = None


NONZERO = 1e-4             # dose threshold used across the project
SPLITS = ("train", "val", "test")

# Effective hyperparameters that change how rewards (or rows) are built. Any key containing
# one of these substrings is part of the reward signature. Extend if the loader changes.
REWARD_KEY_PATTERNS = ("health_score", "reward", "actions_shift", "mortality", "terminal")


# =============================================================================
# Small array helpers
# =============================================================================

def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def states(p) -> np.ndarray:
    return _np(p.rl_dataset["states"])


def next_states(p) -> np.ndarray:
    return _np(p.rl_dataset["next_states"])


def actions(p) -> np.ndarray:
    return _np(p.rl_dataset["actions"]).astype(np.float32)


def next_actions(p) -> np.ndarray:
    return _np(p.rl_dataset["next_actions"]).astype(np.float32)


def rewards(p) -> np.ndarray:
    return _np(p.rl_dataset["rewards"]).reshape(-1).astype(np.float64)


def dones(p) -> np.ndarray:
    return _np(p.rl_dataset["dones"]).reshape(-1).astype(bool)


def mortality(p) -> Optional[np.ndarray]:
    m = p.rl_dataset.get("mortality")
    return None if m is None else _np(m).reshape(-1).astype(bool)


def n_rows(p) -> int:
    return int(p.rl_dataset["states"].shape[0])


def trajectory_ids(p) -> np.ndarray:
    """Integer trajectory id per RL row, from the dones vector."""
    d = dones(p)
    return np.concatenate(([0], np.cumsum(d)[:-1]))


# =============================================================================
# Health score aligned to RL rows
# =============================================================================

def _row_keys(x, decimals=5):
    x = np.ascontiguousarray(np.round(np.asarray(x, dtype=np.float64), decimals))
    x[x == 0] = 0.0
    return [row.tobytes() for row in x]


def rl_row_index(p, which="states") -> np.ndarray:
    """
    Index of the Dynamic-DeepHit row each RL row came from, by matching rl_dataset[which]
    to dynamic_deephit_predictions['model_outputs']. Raises on unmatched/ambiguous rows.
    """
    lat = p.dynamic_deephit_predictions.get("model_outputs")
    if lat is None:
        raise KeyError("dynamic_deephit_predictions has no 'model_outputs' to match RL states against")
    lat, st = _np(lat), _np(p.rl_dataset[which])
    if lat.shape[1] != st.shape[1]:
        raise ValueError(f"model_outputs width {lat.shape[1]} != RL {which} width {st.shape[1]}")
    lookup, dup = {}, set()
    for i, k in enumerate(_row_keys(lat)):
        if k in lookup:
            dup.add(k)
        lookup[k] = i
    keys = _row_keys(st)
    idx = np.array([lookup.get(k, -1) for k in keys])
    n_missing, n_amb = int((idx < 0).sum()), int(sum(k in dup for k in keys))
    if n_missing or n_amb:
        raise ValueError(f"could not align RL {which} to model_outputs: {n_missing} unmatched, "
                         f"{n_amb} ambiguous (are states transformed after loading?)")
    return idx


def health_score(p) -> Tuple[np.ndarray, np.ndarray]:
    """
    (hs, dhs) aligned 1:1 with RL rows: hs[i] = health score at RL state i,
    dhs[i] = HS(next state) - HS(state) (unclipped, unscaled health component of the reward).
    Fast path when the RL dataset keeps every timestep except each stay's last; otherwise
    rows are aligned by matching states/next states to model_outputs. Never silently misaligns.
    """
    hs_full = np.asarray(p.dynamic_deephit_predictions["health_score"], dtype=np.float64)
    ptr = np.asarray(p.dynamic_deephit_predictions["index_ptr"])
    if len(hs_full) - (len(ptr) - 1) == n_rows(p):
        hs, dhs = [], []
        for s, e in zip(ptr[:-1], ptr[1:]):
            traj = hs_full[s:e]
            hs.append(traj[:-1])
            dhs.append(np.diff(traj))
        return np.concatenate(hs), np.concatenate(dhs)
    idx, nidx = rl_row_index(p, "states"), rl_row_index(p, "next_states")
    return hs_full[idx], hs_full[nidx] - hs_full[idx]


def severity_edges(train_hs, n_bins=10) -> np.ndarray:
    """Quantile edges from TRAIN health score (low = sicker); outer edges open."""
    e = np.quantile(np.asarray(train_hs), np.linspace(0, 1, n_bins + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def severity_bin(hs, edges) -> np.ndarray:
    return np.clip(np.digitize(np.asarray(hs), edges[1:-1]), 0, len(edges) - 2)


# =============================================================================
# Processor construction and cache
# =============================================================================

def reward_signature(record: RunRecord) -> Tuple:
    """Sorted (key, value) pairs of the run's reward-related effective hyperparameters."""
    items = [(k, str(v)) for k, v in record.effective.items()
             if any(pat in k.lower() for pat in REWARD_KEY_PATTERNS)]
    return tuple(sorted(items))


def _data_key(record: RunRecord) -> Tuple:
    di = record.data_identity
    if di is None or not di.is_known:
        raise ValueError(f"{record.label}: unknown data identity; cannot build its processor")
    return (di.key, reward_signature(record))


def load_processors(config_manager, hyperparams):
    """
    Train/val/test processors with rewards and RL datasets, exactly as for training.
    (Moved from evaluate_terminal_reward_models.load_data; the building blocks are the
    training pipeline's own: create_processor_from_shards, process_reward, create_rl_dataset.)
    The reward median is fitted on train and applied to val/test; the state scaler is fitted
    on train (None here: states are the raw Dynamic-DeepHit model outputs).
    """
    from src.rl.training import create_processor_from_shards
    train_p, val_p, test_p = create_processor_from_shards(config_manager, normalize=True, set_to_return="all")

    health_score_median = train_p.process_reward(config_manager, hyperparams)
    test_p.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
    val_p.process_reward(config_manager, hyperparams, health_score_median=health_score_median)

    create_rl_datasets(train_p, val_p, test_p, config_manager, hyperparams)
    return train_p, val_p, test_p


def create_rl_datasets(train_p, val_p, test_p, config_manager, hyperparams):
    """
    Build rl_dataset on each processor (moved from evaluate_terminal_reward_models.
    create_test_rl_dataset). Returns the scaler fitted on train.
    """
    common = dict(states="model_outputs",
                  action_names=config_manager.data_config["actions"],
                  use_compressed_latents=hyperparams["use_compressed_latents"],
                  n_step=hyperparams.get("n_step", 1),
                  gamma=hyperparams["discount"],
                  actions_shift=hyperparams.get("actions_shift", True))
    scaler = train_p.create_rl_dataset(scaler=None, **common)
    test_p.create_rl_dataset(scaler=scaler, **common)
    val_p.create_rl_dataset(scaler=scaler, **common)
    return scaler


def build_processors(record: RunRecord, keep: Iterable[str] = ("val",), config_manager=None) -> Dict[str, Any]:
    """
    Run the project's data loader for one run's data identity, using its effective
    hyperparameters and a COPY of its config manager (paths remapped). Returns
    {split: processor} for `keep`. For multi-seed data the config's training seed is set
    to the identity's state seed so the loader selects that Dynamic-DeepHit folder.

    config_manager: a config already prepared for this machine (e.g. LoadedModel.config_manager).
    If None, the run's config is loaded with the training pipeline's load_configuration.
    """
    import copy as _copy
    from .runs import remap_config_manager
    if config_manager is not None:
        cm = remap_config_manager(_copy.deepcopy(config_manager))
    else:
        cm = record.load_config_manager(remap=True, prepared=True)
    if cm is None:
        raise ValueError(f"{record.label}: no config manager found; cannot build processors")
    if "window_length" not in getattr(cm, "training_config", {}):
        # ShardProcessor.__init__ reads training_config['window_length'] unconditionally, but it is a
        # Dynamic-DeepHit setting that the RL processors never use, so RL configs need not have it.
        # Set it to None on OUR COPY only: if any code path does use it, it fails loudly instead of
        # silently using a made-up value.
        cm.training_config["window_length"] = None
    cm.seed_mode = "five_seeds"      # loader must pick the Dynamic-DeepHit folder by seed (RL seed k -> DDH seed k)
    di = record.data_identity
    if di.state_seed is not None:
        cm.training_config["seed"] = di.state_seed
    hp = dict(record.effective)
    tr, va, te = load_processors(cm, hp)
    out = dict(zip(SPLITS, (tr, va, te)))
    for s in SPLITS:
        if s not in keep:
            del out[s]
    del tr, va, te
    gc.collect()
    return out


class DataStore:
    """
    Session cache of processors keyed by (data identity, reward signature).
      processor(model_or_record, split)   processor for that run's own data
      pair(models, split)                 {group: {seed: (model, processor)}}
      processors(models, split)           {group: {seed: processor}}
      reference(split)                    one processor for a split shared by all runs
                                          (e.g. train actions / averaged health score)
    keep: splits kept in memory per identity; other splits are built on request.
    """

    def __init__(self, keep: Iterable[str] = ("val",)):
        self.keep = tuple(keep)
        self._cache: Dict[Tuple, Dict[str, Any]] = {}
        self._labels: Dict[Tuple, str] = {}
        self._reference: Dict[str, Any] = {}
        self._first_record: Optional[RunRecord] = None

    @staticmethod
    def _record(x) -> RunRecord:
        return x.record if hasattr(x, "record") else x

    def processor(self, model_or_record, split="val"):
        rec = self._record(model_or_record)
        cm = getattr(model_or_record, "config_manager", None)
        if self._first_record is None:
            self._first_record = rec
        key = _data_key(rec)
        entry = self._cache.setdefault(key, {})
        if split not in entry:
            want = tuple(sorted(set(self.keep) | {split} | set(entry)))
            print(f"  building processors for {rec.data_identity.short()}  (splits: {', '.join(want)})", flush=True)
            entry.update(build_processors(rec, keep=want, config_manager=cm))
            self._labels[key] = rec.data_identity.short()
        return entry[split]

    def processors(self, models: Dict, split="val") -> Dict:
        return {g: {s: self.processor(m, split) for s, m in seeds.items()} for g, seeds in models.items()}

    def pair(self, models: Dict, split="val") -> Dict:
        return {g: {s: (m, self.processor(m, split)) for s, m in seeds.items()} for g, seeds in models.items()}

    def reference(self, split="train", record: Optional[RunRecord] = None):
        """
        A single processor for `split`, from `record` (default: the first run seen). Use for
        quantities identical across runs, e.g. training actions and the averaged health score.
        Do NOT use its states for any model other than the one it belongs to.
        """
        if split not in self._reference:
            rec = record or self._first_record
            if rec is None:
                raise ValueError("no run seen yet; pass record=")
            self._reference[split] = self.processor(rec, split)
        return self._reference[split]

    def summary(self) -> pd.DataFrame:
        rows = [{"data": self._labels.get(k, "?"), "splits_cached": ", ".join(sorted(v)),
                 "n_val_rows": n_rows(v["val"]) if "val" in v else None} for k, v in self._cache.items()]
        return pd.DataFrame(rows)

    def clear(self):
        self._cache.clear()
        self._reference.clear()
        gc.collect()


# =============================================================================
# Checks
# =============================================================================

def check_rows_aligned(processors: Dict[Any, Any]) -> pd.DataFrame:
    """
    For processors that should describe the same patient-hours (e.g. one per seed):
    same index_ptr / actions are REQUIRED; the table also shows whether the health score,
    rewards and states are identical to the first processor (shared reward -> identical
    rewards; seed-specific states -> states differ).
    """
    items = list(processors.items())
    ref_key, ref = items[0]
    ref_ptr = np.asarray(ref.dynamic_deephit_predictions["index_ptr"])
    ref_act, ref_rew, ref_st = actions(ref), rewards(ref), states(ref)
    ref_hs = np.asarray(ref.dynamic_deephit_predictions["health_score"])
    rows = {}
    for k, p in items:
        st = states(p)
        same_shape = st.shape == ref_st.shape
        rows[k] = {
            "n_rows": n_rows(p),
            "same_index_ptr": np.array_equal(np.asarray(p.dynamic_deephit_predictions["index_ptr"]), ref_ptr),
            "same_actions": same_shape and np.allclose(actions(p), ref_act),
            "same_health_score": np.allclose(np.asarray(p.dynamic_deephit_predictions["health_score"]), ref_hs),
            "same_rewards": same_shape and np.allclose(rewards(p), ref_rew),
            "same_states": same_shape and np.allclose(st, ref_st),
        }
    df = pd.DataFrame(rows).T
    bad = df.index[~(df["same_index_ptr"].astype(bool) & df["same_actions"].astype(bool))].tolist()
    if bad:
        warnings.warn(f"processors not row-aligned with {ref_key}: {bad}")
    return df


def bellman_residual(model, p, gamma: Optional[float] = None, truncate_survival=True) -> Dict[str, float]:
    """
    SARSA-style residual on logged data: |Q(s,a) - (r + gamma * notdone * Q(s',a'))|, with
    a' the logged next action. Terminal handling mirrors training: a trajectory end is
    terminal only if the patient died (survival treated as truncation) when
    truncate_survival=True. Reported relative to sd(Q) so critics on different scales compare.
    A relative check: compare a critic across processors, not against zero.
    """
    g = gamma if gamma is not None else float(model.record.get("discount", 0.975))
    q = model.q(states(p), actions(p))
    q_next = model.q(next_states(p), next_actions(p))
    d = dones(p)
    m = mortality(p)
    terminal = d & m if (truncate_survival and m is not None) else d
    target = rewards(p) + g * (~terminal) * q_next
    res = np.abs(q - target)
    return {"median_abs_residual": float(np.median(res)), "Q_sd": float(q.std()),
            "rel_residual": float(np.median(res) / max(q.std(), 1e-8))}


def bellman_check(models: Dict[Any, Any], processors: Dict[Any, Any], gamma=None) -> pd.DataFrame:
    """
    Rows = models, columns = processors: relative Bellman residual of each model's critic on
    each processor. Each model should be best (lowest) on its OWN data; 'matches_own' uses
    equal keys in the two dicts (e.g. seeds) to say which processor is its own.
    """
    rows = {}
    for mk, m in models.items():
        r = {f"data {pk}": bellman_residual(m, p, gamma)["rel_residual"] for pk, p in processors.items()}
        best = min(processors, key=lambda pk: r[f"data {pk}"])
        r["best_match"] = best
        r["matches_own"] = (best == mk) if mk in processors else None
        rows[f"model {mk}"] = r
    return pd.DataFrame(rows).T