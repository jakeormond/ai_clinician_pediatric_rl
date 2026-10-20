"""
src/rl/analysis/runs.py

Everything known about trained runs BEFORE any model or data is loaded:
  - discover run folders under a directory (single seed, multi-seed, or sweeps)
  - load each run's hyperparameters (results.json / checkpoint) and config manager
  - compute EFFECTIVE hyperparameters: gating rules + recorded overrides
  - resolve each run's DATA IDENTITY (which state embedding and reward it was trained on)
  - group runs into a Study: {group key: {seed: RunRecord}} with consistency checks

Design rules (from past problems):
  * One source of truth per fact. Seed and hyperparameters come from the run's own
    results.json / checkpoint, never from config_manager.pkl, which has been shared or
    overwritten across runs. The pickle is used only for fields that exist nowhere
    else (data paths, seed_mode), and those are cross-checked where possible.
  * Overrides are data, not code. Values that were hardcoded at training time are
    recorded in a run_overrides.yaml next to the runs, applied automatically, printed,
    and stored on the record.
  * Fail loudly on duplicate seeds and on unexpected differences within/between groups.
  * Cheap by default. Reading checkpoints (torch.load) is optional (verify_checkpoint).

Adding a model family: call register_family(...) with a detector and, if needed, custom
readers. Grouping, overrides and data identity work unchanged for every family.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

try:
    import yaml
except ImportError:          # overrides files then have to be JSON
    yaml = None


RUN_MARKERS = ("final_model.pt", "results.json")      # a folder containing either is a run folder
OVERRIDE_FILENAMES = ("run_overrides.yaml", "run_overrides.yml", "run_overrides.json")
CONFIG_PICKLE = "config_manager.pkl"

# keys that legitimately differ between runs of the same group
DEFAULT_IGNORE_KEYS = frozenset({"seed", "wandb_id", "use_fixed_seed", "run_id", "experiment_dir",
                                 "timestamp", "job_id"})


# =============================================================================
# Records
# =============================================================================

@dataclass(frozen=True)
class DataIdentity:
    """
    Which inputs a run was trained on. Two runs with equal identities can share one processor.
      state_source : directory the state embedding came from (Dynamic-DeepHit results dir)
      state_seed   : Dynamic-DeepHit seed whose outputs were used as states
                     (None = single-seed directory, nothing to select)
      reward_source: directory the reward (health score) came from
      reward_seed  : Dynamic-DeepHit seed whose health score was used
                     (None = a single file used by every seed, e.g. the averaged reward)
      seed_mode    : the config's seed_mode
    """
    state_source: Optional[str]
    state_seed: Optional[int]
    reward_source: Optional[str]
    reward_seed: Optional[int]
    seed_mode: Optional[str]

    @property
    def key(self) -> Tuple:
        return (self.state_source, self.state_seed, self.reward_source, self.reward_seed, self.seed_mode)

    @property
    def is_known(self) -> bool:
        return self.state_source is not None and self.reward_source is not None

    def short(self) -> str:
        s = os.path.basename(str(self.state_source).rstrip("/")) if self.state_source else "?"
        r = "shared reward" if self.reward_seed is None else f"reward seed {self.reward_seed}"
        st = "single-seed states" if self.state_seed is None else f"state seed {self.state_seed}"
        return f"{st}, {r} [{s}]"


@dataclass
class RunRecord:
    run_dir: str
    family: str
    seed: Optional[int]
    hyperparams: Dict[str, Any]                 # as saved by the run
    effective: Dict[str, Any]                   # after gating rules and overrides
    overrides_applied: Dict[str, Tuple[Any, Any, str]] = field(default_factory=dict)   # key -> (saved, used, reason)
    data_identity: Optional[DataIdentity] = None
    checkpoint_conflicts: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)     # key -> (results.json, checkpoint)
    config_path: Optional[str] = None
    label: str = ""

    def get(self, key, default=None):
        """Effective value of a hyperparameter."""
        return self.effective.get(key, default)

    def load_config_manager(self, remap=True, prepared=True, quiet=True):
        """
        The run's config manager, or None.
          prepared=True : loaded with the training pipeline's src.config_manager.load_configuration,
                          which sets paths/environment for THIS machine (what the data loader
                          expects); prepared=False returns the raw pickle.
          remap=True    : then apply PATH_MAPPINGS to every path-like attribute.
          quiet=True    : silence what load_configuration prints.
        Use only for fields not stored elsewhere (seed and hyperparameters come from the run).
        """
        if not self.config_path:
            return None
        if prepared:
            import contextlib, io
            from src.config_manager import load_configuration
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    cm = load_configuration(self.config_path)
            else:
                cm = load_configuration(self.config_path)
        else:
            cm = _load_pickle(self.config_path)
        return remap_config_manager(cm) if remap else cm

    def __repr__(self):
        return f"RunRecord({self.label or self.run_dir})"


# =============================================================================
# Model families (extension point)
# =============================================================================

@dataclass
class FamilySpec:
    """
    How to recognise and read one model family's run folders.
      detect(run_dir)            -> bool, used when discover_runs(family=None)
      read_hyperparams(run_dir)  -> dict of saved hyperparameters
      read_seed(run_dir, hp)     -> int or None
      data_identity(record, cm)  -> DataIdentity (cm = loaded config manager or None)
    Defaults suit runs saved by the project's training scripts (results.json with a
    'hyperparameters' dict, optional final_model.pt with 'hyperparameters').
    """
    name: str
    detect: Callable[[str], bool]
    read_hyperparams: Optional[Callable[[str], Dict[str, Any]]] = None
    read_seed: Optional[Callable[[str, Dict[str, Any]], Optional[int]]] = None
    data_identity: Optional[Callable[["RunRecord", Any], DataIdentity]] = None


FAMILIES: Dict[str, FamilySpec] = {}


def register_family(spec: FamilySpec):
    """Register (or replace) a model family."""
    FAMILIES[spec.name] = spec
    return spec


def _path_contains(*parts):
    return lambda run_dir: any(f"{os.sep}{p}{os.sep}" in os.path.abspath(run_dir) + os.sep for p in parts)


# =============================================================================
# Path mapping: runs trained on the cluster, analysed locally (or vice versa)
# =============================================================================
#
# Configs record absolute paths from the machine a run was trained on. PATH_MAPPINGS is a
# list of (from_prefix, to_prefix) pairs applied to every path read from a run's config
# (data identity, config manager attributes). Longest matching prefix wins. Set once per
# session with set_path_mappings(...) or load_path_mappings(file); empty = no remapping.

PATH_MAPPINGS: List[Tuple[str, str]] = []


def set_path_mappings(mappings: Iterable[Tuple[str, str]]):
    """e.g. set_path_mappings([("/rds/general/user/jormond/home/projects/ai_clinician",
                                 "/home/jake/Documents/AI_Clinician/models")])"""
    global PATH_MAPPINGS
    PATH_MAPPINGS = sorted(((a.rstrip("/"), b.rstrip("/")) for a, b in mappings),
                           key=lambda ab: len(ab[0]), reverse=True)
    return PATH_MAPPINGS


def load_path_mappings(path):
    """Read mappings from a YAML/JSON file:  mappings: [[from_prefix, to_prefix], ...]"""
    if path.endswith(".json"):
        data = _load_json_early(path)
    else:
        if yaml is None:
            raise ImportError("PyYAML needed for YAML path-mapping files")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    return set_path_mappings([tuple(m) for m in data.get("mappings", [])])


def _load_json_early(path):
    with open(path) as f:
        return json.load(f)


def map_path(p):
    """Apply PATH_MAPPINGS to one path (unchanged if no prefix matches or p is not a string)."""
    if not isinstance(p, str):
        return p
    for a, b in PATH_MAPPINGS:
        if p == a or p.startswith(a + "/"):
            return b + p[len(a):]
    return p


def remap_paths(obj):
    """Recursively apply map_path to strings inside dicts / lists / tuples."""
    if isinstance(obj, str):
        return map_path(obj)
    if isinstance(obj, dict):
        return {k: remap_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_paths(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(remap_paths(v) for v in obj)
    return obj


def remap_config_manager(cm):
    """
    Copy of a config manager with every path-like string attribute remapped
    (data_config, paths, training_config, plain string attributes). Properties computed
    from these attributes (e.g. get_data_dir) then resolve to local paths too.
    """
    import copy as _copy
    if cm is None or not PATH_MAPPINGS:
        return cm
    cm2 = _copy.deepcopy(cm)
    for attr, val in list(vars(cm2).items()):
        if isinstance(val, (str, dict, list, tuple)):
            setattr(cm2, attr, remap_paths(val))
    return cm2


# =============================================================================
# Low-level readers
# =============================================================================

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_results_json(run_dir) -> Dict[str, Any]:
    path = os.path.join(run_dir, "results.json")
    return _load_json(path) if os.path.isfile(path) else {}


def default_read_hyperparams(run_dir) -> Dict[str, Any]:
    """results.json['hyperparameters'] (falls back to an empty dict)."""
    return dict(read_results_json(run_dir).get("hyperparameters", {}))


def read_checkpoint_hyperparams(run_dir) -> Optional[Dict[str, Any]]:
    """final_model.pt['hyperparameters'], or None. Imports torch lazily (slow for many runs)."""
    path = os.path.join(run_dir, "final_model.pt")
    if not os.path.isfile(path):
        return None
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyperparameters") if isinstance(ckpt, dict) else None
    return dict(hp) if hp is not None else None


def default_read_seed(run_dir, hp) -> Optional[int]:
    """Seed from the run's own hyperparameters, then results.json top level."""
    for src in (hp, read_results_json(run_dir)):
        s = src.get("seed") if isinstance(src, dict) else None
        if s is not None:
            return int(s)
    return None


def find_config_pickle(run_dir, max_up=1) -> Optional[str]:
    """config_manager.pkl in the run folder, or up to `max_up` parent folders."""
    d = run_dir
    for _ in range(max_up + 1):
        p = os.path.join(d, CONFIG_PICKLE)
        if os.path.isfile(p):
            return p
        d = os.path.dirname(d)
    return None


def find_run_dirs(root) -> List[str]:
    """Every folder under root (inclusive) that contains a run marker file."""
    out = []
    for dirpath, _, files in os.walk(root):
        if any(m in files for m in RUN_MARKERS):
            out.append(dirpath)
    return sorted(out)


# =============================================================================
# Effective hyperparameters: gating rules + overrides
# =============================================================================

# Each rule: (name, fn(hp) -> dict of {key: (new_value, reason)}). Applied in order.
EFFECTIVE_RULES: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Tuple[Any, str]]]]] = []


def register_rule(name, fn):
    EFFECTIVE_RULES.append((name, fn))


def _vaso_reward_gate(hp):
    """vaso_reward_coefficient only takes effect when add_reward_to_vaso_dosing is True."""
    if "vaso_reward_coefficient" in hp and not hp.get("add_reward_to_vaso_dosing", False):
        if hp.get("vaso_reward_coefficient") not in (0, 0.0, None):
            return {"vaso_reward_coefficient": (0.0, "add_reward_to_vaso_dosing is False")}
    return {}


register_rule("vaso_reward_gate", _vaso_reward_gate)


def _read_override_file(path) -> List[dict]:
    if path.endswith(".json"):
        data = _load_json(path)
    else:
        if yaml is None:
            raise ImportError(f"PyYAML needed to read {path}; use run_overrides.json instead")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    entries = data.get("overrides", []) if isinstance(data, dict) else data
    for e in entries:
        if "set" not in e or "reason" not in e:
            raise ValueError(f"{path}: every override needs 'set' and 'reason' (got {e})")
    return entries


def find_override_files(run_dir, stop_at=None) -> List[str]:
    """Override files from the run folder up to stop_at (inclusive), ordered outermost first."""
    found, d = [], os.path.abspath(run_dir)
    stop = os.path.abspath(stop_at) if stop_at else None
    while True:
        for name in OVERRIDE_FILENAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                found.append(p)
        if d == stop or os.path.dirname(d) == d:
            break
        d = os.path.dirname(d)
    return found[::-1]


def _matches(hp, match: Optional[dict]) -> bool:
    return all(str(hp.get(k)) == str(v) for k, v in (match or {}).items())


def compute_effective(hp, run_dir, stop_at=None):
    """
    Effective hyperparameters = saved hyperparameters, then gating rules, then overrides
    from run_overrides.* files (outermost first, so files closer to the run win).

    Override file format (YAML or JSON):
        overrides:
          - set: {fluid_mean_reg_scale: 0.05, vaso_mean_reg_scale: 0.05}
            reason: "hardcoded in get_trainer for these runs"
            match: {seed: 0}        # optional: only runs whose SAVED values match
    Returns (effective, applied) with applied = {key: (saved, used, reason)}.
    """
    eff = dict(hp)
    applied = {}
    for name, rule in EFFECTIVE_RULES:
        for k, (v, why) in rule(eff).items():
            applied[k] = (hp.get(k, "MISSING"), v, f"rule {name}: {why}")
            eff[k] = v
    for path in find_override_files(run_dir, stop_at):
        for e in _read_override_file(path):
            if _matches(hp, e.get("match")):
                for k, v in e["set"].items():
                    applied[k] = (hp.get(k, "MISSING"), v, f"{os.path.basename(path)}: {e['reason']}")
                    eff[k] = v
    return eff, applied


# =============================================================================
# Data identity
# =============================================================================

def _has_direct_predictions(directory) -> bool:
    """True if a predictions file sits directly in this folder (a shared/averaged reward file)."""
    if not directory or not os.path.isdir(directory):
        return False
    return any(f.endswith("_predictions.h5") for f in os.listdir(directory))


def default_data_identity(record: "RunRecord", cm) -> DataIdentity:
    """
    Mirrors the data processor's selection rules:
      seed_mode == 'single_seed' -> the single Dynamic-DeepHit folder; no seed selection
      otherwise                  -> the folder whose seed matches the run's training seed
    A reward directory with a predictions file directly inside it (the averaged reward)
    is shared by every seed (reward_seed None).
    """
    if cm is None:
        return DataIdentity(None, None, None, None, None)
    dc = getattr(cm, "data_config", {}) or {}
    state_src, reward_src = map_path(dc.get("state_path")), map_path(dc.get("reward_path"))
    for name, p in (("state_path", state_src), ("reward_path", reward_src)):
        if p is not None and not os.path.isdir(p):
            warnings.warn(f"{record.run_dir}: {name} {p} does not exist on this machine "
                          f"(missing path mapping? see set_path_mappings)")
    # RL seed k always uses Dynamic-DeepHit seed k (states are per seed); the config's seed_mode is ignored.
    state_seed = record.seed
    reward_seed = None if _has_direct_predictions(reward_src) else record.seed      # averaged reward is shared
    return DataIdentity(state_src, state_seed, reward_src, reward_seed, "matched_seeds")


# =============================================================================
# Building records
# =============================================================================

def detect_family(run_dir) -> Optional[str]:
    hits = [name for name, spec in FAMILIES.items() if spec.detect(run_dir)]
    if len(hits) > 1:
        raise ValueError(f"{run_dir} matches several families {hits}; pass family= explicitly")
    return hits[0] if hits else None


def load_run(run_dir, family=None, overrides_root=None, verify_checkpoint=False) -> RunRecord:
    """Build the RunRecord for one run folder."""
    family = family or detect_family(run_dir)
    if family is None:
        raise ValueError(f"cannot detect the model family of {run_dir}; pass family=")
    spec = FAMILIES[family]

    hp = (spec.read_hyperparams or default_read_hyperparams)(run_dir)
    conflicts = {}
    if verify_checkpoint:
        ck = read_checkpoint_hyperparams(run_dir)
        if ck is not None:
            for k in set(hp) | set(ck):
                if k in hp and k in ck and str(hp[k]) != str(ck[k]):
                    conflicts[k] = (hp[k], ck[k])
                elif k not in hp:
                    hp[k] = ck[k]              # fill gaps from the checkpoint
            if conflicts:
                warnings.warn(f"{run_dir}: results.json and checkpoint disagree on {sorted(conflicts)}")

    seed = (spec.read_seed or default_read_seed)(run_dir, hp)
    eff, applied = compute_effective(hp, run_dir, stop_at=overrides_root)
    rec = RunRecord(run_dir=run_dir, family=family, seed=seed, hyperparams=hp, effective=eff,
                    overrides_applied=applied, checkpoint_conflicts=conflicts,
                    config_path=find_config_pickle(run_dir))

    cm = None
    if rec.config_path:
        try:
            cm = _load_pickle(rec.config_path)
        except Exception as e:                       # class definitions missing, etc.
            warnings.warn(f"{run_dir}: could not read {CONFIG_PICKLE}: {e}")
    rec.data_identity = (spec.data_identity or default_data_identity)(rec, cm)
    rec.label = f"{family} seed {seed}"
    return rec


def discover_runs(root, family=None, verify_checkpoint=False, overrides_root=None,
                  include: Optional[Callable[[RunRecord], bool]] = None) -> List[RunRecord]:
    """
    All runs under root as RunRecords. family=None detects each run's family.
    overrides_root: highest folder searched for run_overrides files (default: root).
    include: optional filter on records.
    """
    overrides_root = overrides_root or root
    recs = []
    for d in find_run_dirs(root):
        r = load_run(d, family=family, overrides_root=overrides_root, verify_checkpoint=verify_checkpoint)
        if include is None or include(r):
            recs.append(r)
    if not recs:
        warnings.warn(f"no runs found under {root}")
    report_overrides(recs, header=f"overrides applied under {root}")
    return recs


def report_overrides(records, header="overrides applied"):
    """Print each distinct override once, with the seeds it applied to."""
    seen: Dict[Tuple, List] = {}
    for r in records:
        for k, (saved, used, why) in r.overrides_applied.items():
            if str(saved) != str(used):
                seen.setdefault((k, str(saved), str(used), why), []).append(r.seed)
    if seen:
        print(f"  {header}:")
        for (k, saved, used, why), seeds in seen.items():
            print(f"    {k}: saved={saved} -> used={used}  [{why}]  seeds={sorted(seeds, key=lambda x: (x is None, x))}")


# =============================================================================
# Studies: {group key: {seed: RunRecord}}
# =============================================================================

def _fmt_key(group_by, key):
    if not group_by:
        return "all"
    return ", ".join(str(v) if k == "_group" else f"{k}={v}" for k, v in zip(group_by, key))


class Study:
    """
    A set of runs grouped by one or more hyperparameters (effective values):
      single seed : one group, one seed
      multi-seed  : one group, several seeds        (group_by = ())
      sweep       : one group per swept value, each with one or more seeds
    Iterate with .items() -> (group_key, seed, record). Duplicate seeds within a group raise.
    """

    def __init__(self, runs: Iterable[RunRecord], group_by: Tuple[str, ...] = (), name: str = ""):
        self.name = name
        self.group_by = tuple(group_by)
        self.groups: Dict[Tuple, Dict[Optional[int], RunRecord]] = {}
        for r in runs:
            key = tuple(r.get(k) for k in self.group_by)
            g = self.groups.setdefault(key, {})
            if r.seed in g:
                raise ValueError(f"study {name!r}: seed {r.seed} appears twice in group "
                                 f"{_fmt_key(self.group_by, key)}:\n  {g[r.seed].run_dir}\n  {r.run_dir}")
            g[r.seed] = r
            grp = _fmt_key(self.group_by, key) if self.group_by else ""
            r.label = " ".join(x for x in (name, grp, f"seed {r.seed}") if x)
        self.groups = {k: dict(sorted(v.items(), key=lambda kv: (kv[0] is None, kv[0])))
                       for k, v in sorted(self.groups.items(), key=lambda kv: tuple(str(x) for x in kv[0]))}

    # ---- access ----
    def items(self) -> Iterator[Tuple[Tuple, Optional[int], RunRecord]]:
        for key, seeds in self.groups.items():
            for s, r in seeds.items():
                yield key, s, r

    def runs(self) -> List[RunRecord]:
        return [r for _, _, r in self.items()]

    def group(self, *key) -> Dict[Optional[int], RunRecord]:
        """Runs of one group by its key values, e.g. study.group(0.022)."""
        return self.groups[tuple(key)]

    @property
    def keys(self) -> List[Tuple]:
        return list(self.groups)

    def seeds(self, common=True) -> List[Optional[int]]:
        """Seeds present in every group (common=True) or in any group."""
        sets = [set(g) for g in self.groups.values()]
        if not sets:
            return []
        s = set.intersection(*sets) if common else set.union(*sets)
        return sorted(s, key=lambda x: (x is None, x))

    def data_identities(self) -> Dict[Tuple, DataIdentity]:
        """Distinct data identities used by the study (one processor needed per identity)."""
        return {r.data_identity.key: r.data_identity for r in self.runs() if r.data_identity is not None}

    def subset(self, seeds=None, keys=None) -> "Study":
        if keys is not None:
            keys = [tuple(k) if isinstance(k, (tuple, list)) else (k,) for k in keys]
        runs = [r for k, s, r in self.items()
                if (seeds is None or s in seeds) and (keys is None or k in keys)]
        return Study(runs, self.group_by, self.name)

    # ---- tables and checks ----
    def table(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """One row per run: group key, seed, selected effective hyperparameters, data identity."""
        rows = []
        for key, s, r in self.items():
            row = {"group": _fmt_key(self.group_by, key), "seed": s, "family": r.family}
            for c in (columns or []):
                row[c] = r.get(c)
            row["n_overrides"] = sum(str(a) != str(b) for a, b, _ in r.overrides_applied.values())
            row["data"] = r.data_identity.short() if r.data_identity else "?"
            row["run_dir"] = r.run_dir
            rows.append(row)
        return pd.DataFrame(rows).set_index(["group", "seed"])

    def differences(self, ignore=DEFAULT_IGNORE_KEYS) -> Dict[str, Dict[str, Any]]:
        """
        Effective hyperparameters that differ (a) across seeds within each group and
        (b) between groups (lowest seed of each). Within-group differences are unexpected;
        between-group differences should be exactly the group_by keys.
        """
        out = {}
        for key, seeds in self.groups.items():
            hps = {s: r.effective for s, r in seeds.items()}
            allk = sorted(set().union(*hps.values())) if hps else []
            out[f"within {_fmt_key(self.group_by, key)}"] = {
                k: {s: hps[s].get(k) for s in hps} for k in allk
                if k not in ignore and len({str(hps[s].get(k)) for s in hps}) > 1}
        firsts = {_fmt_key(self.group_by, k): next(iter(g.values())).effective for k, g in self.groups.items()}
        allk = sorted(set().union(*firsts.values())) if firsts else []
        out["between groups"] = {k: {g: firsts[g].get(k) for g in firsts} for k in allk
                                 if k not in ignore and len({str(firsts[g].get(k)) for g in firsts}) > 1}
        return out

    def check(self, expected_between: Optional[Iterable[str]] = None, ignore=DEFAULT_IGNORE_KEYS,
              require_same_seeds=True, strict=False) -> Dict[str, Any]:
        """
        Consistency checks, printed. expected_between: keys allowed to differ between groups
        (default: group_by). strict=True raises on any problem.
        Problems: within-group differences, unexpected between-group differences, groups
        with different seed sets, runs with unknown data identity, results/checkpoint conflicts.
        """
        expected = set(expected_between if expected_between is not None else self.group_by)
        diffs = self.differences(ignore)
        problems = []
        for name, d in diffs.items():
            if name.startswith("within") and d:
                problems.append(f"{name}: {sorted(d)}")
        unexpected = sorted(set(diffs["between groups"]) - expected)
        if unexpected:
            problems.append(f"unexpected differences between groups: {unexpected}")
        missing = sorted(expected - set(diffs["between groups"])) if len(self.groups) > 1 else []
        if missing:
            problems.append(f"expected to differ between groups but identical: {missing}")
        if require_same_seeds and len({tuple(g) for g in self.groups.values()}) > 1:
            problems.append(f"groups have different seed sets: { {_fmt_key(self.group_by, k): list(g) for k, g in self.groups.items()} }")
        unknown = [r.label for r in self.runs() if r.data_identity is None or not r.data_identity.is_known]
        if unknown:
            problems.append(f"unknown data identity (no readable config): {unknown}")
        conflicted = {r.label: r.checkpoint_conflicts for r in self.runs() if r.checkpoint_conflicts}
        if conflicted:
            problems.append(f"results.json vs checkpoint conflicts: {conflicted}")

        print(f"Study {self.name!r}: {len(self.groups)} group(s), {len(self.runs())} run(s), "
              f"seeds {self.seeds(common=False)}")
        for name, d in diffs.items():
            if d:
                print(f"  {name}:")
                for k, v in d.items():
                    print(f"      {k:40s} {v}")
        if problems:
            print("  PROBLEMS:")
            for p in problems:
                print(f"    - {p}")
            if strict:
                raise ValueError(f"study {self.name!r} failed checks: {problems}")
        else:
            print("  OK")
        return {"differences": diffs, "problems": problems}

    def __repr__(self):
        return f"Study({self.name!r}, groups={len(self.groups)}, runs={len(self.runs())}, group_by={self.group_by})"


def make_study(roots, family=None, group_by=(), name="", verify_checkpoint=False,
               include: Optional[Callable[[RunRecord], bool]] = None, overrides_root=None) -> Study:
    """
    Convenience: discover runs under one or several roots and group them.
      roots: a directory, or {group label: directory} to name groups by folder instead of
             by a hyperparameter (the label is stored as effective['_group'] and used as group_by).
      overrides_root: highest folder searched for run_overrides files (default: each root).
    """
    if isinstance(roots, dict):
        runs = []
        for lbl, d in roots.items():
            for r in discover_runs(d, family=family, verify_checkpoint=verify_checkpoint, include=include,
                                   overrides_root=overrides_root):
                r.effective["_group"] = lbl
                runs.append(r)
        return Study(runs, group_by=("_group",) + tuple(group_by), name=name)
    runs = discover_runs(roots, family=family, verify_checkpoint=verify_checkpoint, include=include,
                         overrides_root=overrides_root)
    return Study(runs, group_by=group_by, name=name)


# =============================================================================
# Built-in families
# =============================================================================

register_family(FamilySpec(name="cql", detect=_path_contains("cql_with_logprobs", "cql")))
register_family(FamilySpec(name="sarsa", detect=lambda d: _path_contains("basic_q")(d) and not _path_contains("fqe")(d)))
register_family(FamilySpec(name="fqe", detect=_path_contains("fqe")))