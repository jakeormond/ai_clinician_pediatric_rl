"""
src/rl/analysis/models.py

Loads trained models from RunRecords (see runs.py) into one standard object, LoadedModel,
whatever the model family. Analyses work with LoadedModel and never with family-specific
loading code.

  model = load_model(record)                 # one run
  models = load_study_models(study)          # {group_key: {seed: LoadedModel}}

LoadedModel gives every family the same interface:
  .record            the RunRecord (hyperparameters, effective values, data identity, ...)
  .actor / .critic   torch modules (None if the family has no such component)
  .components        dict of every loaded module (actor, critic, critic_target, ...)
  .reward_scale      divisor converting Q-values back to health-score points
  .q(states, actions)          critic Q-values (mean of twin heads by default), numpy
  .actor_outputs(states)       family-specific actor outputs (CQL: gate probs, raw means, stds, actions)

Adding a family: write a loader fn(record, state_dim, device) -> LoadedModel and call
register_loader("<family>", fn). Put the config manager the loader prepares in
loader_extras["config_manager"] so data.py can build processors from it. If its actor has a different output API, also pass
actor_outputs=fn(model, states_tensor) -> dict of tensors.

Notes
  * Seeds are cross-checked against the RunRecord; a mismatch raises.
  * Models are cached by run folder, so the same run is only loaded once per session.
  * After editing a model class (e.g. cql_with_logprobs.py) restart the kernel: autoreload
    re-creates the classes and breaks isinstance/super() for already-imported modules.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from .runs import RunRecord, Study


# =============================================================================
# Standard model object
# =============================================================================

@dataclass
class LoadedModel:
    record: RunRecord
    components: Dict[str, Any] = field(default_factory=dict)
    loader_extras: Dict[str, Any] = field(default_factory=dict)     # anything the family loader returned besides modules
    device: Any = None
    _actor_outputs_fn: Optional[Callable] = None

    @property
    def actor(self):
        return self.components.get("actor")

    @property
    def critic(self):
        return self.components.get("critic")

    @property
    def config_manager(self):
        """
        Config manager as prepared by the family loader (environment set up for this
        machine, settings such as window_length filled in). data.py builds processors from
        this; None if the loader did not provide one.
        """
        return self.loader_extras.get("config_manager")

    @property
    def family(self) -> str:
        return self.record.family

    @property
    def seed(self):
        return self.record.seed

    @property
    def reward_scale(self) -> float:
        """Divisor converting Q-values to health-score points (total_health_score_scale, else 1)."""
        s = self.record.get("total_health_score_scale")
        if s is None:
            warnings.warn(f"{self.record.label}: no total_health_score_scale; Q-values left in raw units")
            return 1.0
        return float(s)

    def eval(self):
        for m in self.components.values():
            if hasattr(m, "eval"):
                m.eval()
        return self

    # ---- batched inference -------------------------------------------------
    @torch.no_grad()
    def q(self, states, actions, reduction="mean", batch_size=4096, scaled=False) -> np.ndarray:
        """
        Critic Q(s, a) for numpy/tensor states and actions. reduction: 'mean' | 'min' | 'q1' | 'q2'
        for twin critics (ignored for single-head critics). scaled=True divides by reward_scale
        (health-score points).
        """
        if self.critic is None:
            raise ValueError(f"{self.record.label} has no critic")
        self.critic.eval()
        states = _as_np(states)
        actions = _as_np(actions).astype(np.float32)
        out = []
        for i in range(0, len(states), batch_size):
            s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=self.device)
            a = torch.as_tensor(actions[i:i + batch_size], dtype=torch.float32, device=self.device)
            res = self.critic(s, a)
            if isinstance(res, (tuple, list)):
                q1, q2 = res[0], res[1]
                q = {"mean": (q1 + q2) / 2, "min": torch.min(q1, q2), "q1": q1, "q2": q2}[reduction]
            else:
                q = res
            out.append(q.reshape(-1).cpu())
        q = torch.cat(out).numpy().astype(np.float64)
        return q / self.reward_scale if scaled else q

    @torch.no_grad()
    def actor_outputs(self, states, batch_size=4096) -> Dict[str, np.ndarray]:
        """Family-specific actor outputs for every state, as numpy arrays."""
        if self.actor is None:
            raise ValueError(f"{self.record.label} has no actor")
        if self._actor_outputs_fn is None:
            raise NotImplementedError(f"no actor_outputs function registered for family {self.family!r}")
        self.actor.eval()
        states = _as_np(states)
        chunks: Dict[str, list] = {}
        for i in range(0, len(states), batch_size):
            s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=self.device)
            for k, v in self._actor_outputs_fn(self, s).items():
                chunks.setdefault(k, []).append(v.detach().cpu())
        return {k: torch.cat(v).numpy() for k, v in chunks.items()}

    def __repr__(self):
        comps = ", ".join(self.components)
        return f"LoadedModel({self.record.label}: {comps})"


def _as_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# =============================================================================
# Family loaders (extension point)
# =============================================================================

@dataclass
class LoaderSpec:
    load: Callable[[RunRecord, int, Any], LoadedModel]
    actor_outputs: Optional[Callable[[LoadedModel, torch.Tensor], Dict[str, torch.Tensor]]] = None


LOADERS: Dict[str, LoaderSpec] = {}


def register_loader(family: str, load, actor_outputs=None):
    LOADERS[family] = LoaderSpec(load=load, actor_outputs=actor_outputs)


# =============================================================================
# State dimension from the run's Dynamic-DeepHit model
# =============================================================================

def _ddh_seed_folders(ddh_dir) -> Dict[Optional[int], str]:
    out = {}
    if not ddh_dir or not os.path.isdir(ddh_dir):
        return out
    for f in sorted(os.listdir(ddh_dir)):
        if f.startswith(("unknown", "local")):
            rj = os.path.join(ddh_dir, f, "results.json")
            seed = None
            if os.path.isfile(rj):
                with open(rj) as fh:
                    seed = json.load(fh).get("hyperparameters", {}).get("seed")
            out[seed] = os.path.join(ddh_dir, f)
    return out


def infer_state_dim(record: RunRecord) -> int:
    """
    d_model of the Dynamic-DeepHit model whose outputs are this run's states
    (seed-matched folder for multi-seed runs, the single folder otherwise).
    """
    di = record.data_identity
    if di is None or di.state_source is None:
        raise ValueError(f"{record.label}: unknown state source; pass state_dim explicitly")
    folders = _ddh_seed_folders(di.state_source)
    if di.state_seed is not None:
        if di.state_seed not in folders:
            raise ValueError(f"{record.label}: no Dynamic-DeepHit folder with seed {di.state_seed} in {di.state_source}")
        folder = folders[di.state_seed]
    elif len(folders) == 1:
        folder = next(iter(folders.values()))
    elif os.path.isfile(os.path.join(di.state_source, "results.json")):
        folder = di.state_source
    else:
        raise ValueError(f"{record.label}: cannot pick a Dynamic-DeepHit folder in {di.state_source} "
                         f"(found seeds {sorted(folders, key=str)}); pass state_dim explicitly")
    with open(os.path.join(folder, "results.json")) as f:
        return int(json.load(f)["hyperparameters"]["d_model"])


# =============================================================================
# Public API
# =============================================================================

_CACHE: Dict[str, LoadedModel] = {}


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(record: RunRecord, state_dim: Optional[int] = None, device=None, use_cache=True,
               quiet=True) -> LoadedModel:
    """
    Load one run. state_dim defaults to the run's Dynamic-DeepHit d_model.
    quiet=True suppresses what the family loader prints (config-manager setup etc.);
    warnings and errors still come through.
    """
    if use_cache and record.run_dir in _CACHE:
        return _CACHE[record.run_dir]
    if record.family not in LOADERS:
        raise ValueError(f"no loader registered for family {record.family!r} (register_loader)")
    spec = LOADERS[record.family]
    state_dim = state_dim if state_dim is not None else infer_state_dim(record)
    device = device or default_device()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            model = spec.load(record, state_dim, device)
    else:
        model = spec.load(record, state_dim, device)
    model.device = device
    model._actor_outputs_fn = spec.actor_outputs
    model.eval()
    if use_cache:
        _CACHE[record.run_dir] = model
    return model


def load_study_models(study: Study, state_dim: Optional[int] = None, device=None,
                      verbose=True, quiet=True) -> Dict[tuple, Dict[Any, LoadedModel]]:
    """{group_key: {seed: LoadedModel}} for every run in a study."""
    out: Dict[tuple, Dict[Any, LoadedModel]] = {}
    for key, seed, rec in study.items():
        m = load_model(rec, state_dim=state_dim, device=device, quiet=quiet)
        out.setdefault(key, {})[seed] = m
        if verbose:
            print(f"  loaded {m}")
    return out


def clear_cache():
    _CACHE.clear()


def _check_seed(record: RunRecord, loaded_hp: Dict[str, Any]):
    s = loaded_hp.get("seed") if isinstance(loaded_hp, dict) else None
    if s is not None and record.seed is not None and int(s) != int(record.seed):
        raise ValueError(f"{record.run_dir}: loader reports seed {s} but the run record says {record.seed}")


# =============================================================================
# Built-in families
# =============================================================================
# Loaders build the networks from the model classes used in training and load the run's own
# checkpoint. Architecture comes from the SAVED hyperparameters (record.hyperparams), never
# from effective values, so overrides can never change what is loaded. The config manager is
# the run's, prepared for this machine by the training pipeline's load_configuration.

def _read_checkpoint(record: RunRecord, device):
    path = os.path.join(record.run_dir, "final_model.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{record.label}: no final_model.pt in {record.run_dir}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ck_hp = ckpt.get("hyperparameters") if isinstance(ckpt, dict) else None
    if ck_hp:
        _check_seed(record, ck_hp)
    return ckpt


def _require(record: RunRecord, key):
    if key not in record.hyperparams:
        raise KeyError(f"{record.label}: saved hyperparameters have no {key!r}")
    return record.hyperparams[key]


def _load_cql(record: RunRecord, state_dim: int, device) -> LoadedModel:
    from src.rl.models.cql_with_logprobs import Actor as CQLActor, Critic as CQLCritic
    ckpt = _read_checkpoint(record, device)
    actor = CQLActor(state_dim=state_dim, hidden_dim=_require(record, "actor_hidden_dim")).to(device)
    critic = CQLCritic(state_dim=state_dim, action_dim=2, hidden_dim=_require(record, "critic_hidden_dim")).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    critic.load_state_dict(ckpt["critic_state_dict"])
    return LoadedModel(record=record, components={"actor": actor, "critic": critic},
                       loader_extras={"config_manager": record.load_config_manager()})


def _cql_actor_outputs(model: LoadedModel, s: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    CQL hurdle actor: gate probabilities, raw (pre-sigmoid) magnitude means, stds, and the
    deterministic action exactly as Actor.forward(deterministic=True) computes it.
    """
    actions, _, _, dist = model.actor(s, deterministic=True, return_dist=True)
    f_p, v_p, f_raw, v_raw, f_std, v_std = dist
    return {"actions": actions,
            "f_p": f_p.squeeze(-1), "v_p": v_p.squeeze(-1),
            "f_raw_mean": f_raw.squeeze(-1), "v_raw_mean": v_raw.squeeze(-1),
            "f_std": f_std.squeeze(-1), "v_std": v_std.squeeze(-1)}


def _load_sarsa(record: RunRecord, state_dim: int, device) -> LoadedModel:
    from src.rl.models.basic_q import Critic as SARSACritic
    ckpt = _read_checkpoint(record, device)
    critic = SARSACritic(state_dim, 2, hidden_dim=_require(record, "hidden_dim")).to(device)
    critic.load_state_dict(ckpt["critic_state_dict"])
    return LoadedModel(record=record, components={"critic": critic},
                       loader_extras={"config_manager": record.load_config_manager()})


register_loader("cql", _load_cql, actor_outputs=_cql_actor_outputs)
register_loader("sarsa", _load_sarsa)