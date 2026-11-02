"""
src/rl/analysis/model_locations.py

Where each model type's runs live, as a function of the choices a notebook makes
(model type, hospital, cohort, ...). The folder names are data (HOSPITALS, the builders
below); a notebook only states its choices:

    import src.rl.analysis.model_locations as loc
    sweep_dir = loc.sweep_dir(models_root, model_type="dynamic_deephit", hospital="mimiciv",
                              action_inclusion="with_actions", purpose="for_rewards")

Invalid combinations raise with the valid options, so a typo is caught here, not as a
missing folder later. To add a hospital, a cohort or a model type, edit the tables below.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

# hospital -> the folder naming the action variables, and cohort -> health-score group folder
HOSPITALS: Dict[str, dict] = {
    "choc": dict(
        actions="intake-vis",
        cohorts={"picu": "picu_patients_health_score_from_picu_dataset",
                 "sepsis_picu": "picu_patients_health_score_from_restricted_dataset"},
    ),
    "mimic_iv": dict(
        actions="combined_hourly_iv-vasoactive_inotropic_score",
        cohorts={"icu": "icu_patients_health_score_from_icu_dataset",
                 "sepsis_icu": "icu_patients_health_score_from_restricted_dataset"},
    ),
}

HOSPITAL_ALIASES = {"mimiciv": "mimic_iv"}          # accepted spellings -> folder name on disk
REWARD_TYPES = ("health_score_change_only", "terminal_only", "terminal_bellman")   # RL models only
# model type -> folder name under rl_models/<hospital>/ (the cql folder is spelled log_probs on disk; others to confirm)
RL_FOLDERS = {"basic_q": "basic_q", "cql_with_logprobs": "cql_with_log_probs", "fqe": "fqe",
              "bc_with_logprobs": "bc_with_log_probs"}

ACTION_INCLUSION = ("with_actions", "without_actions")
DDH_PURPOSES = ("for_rewards", "for_states")        # only for Dynamic-DeepHit trained with actions
RL_MODEL_TYPES = ("basic_q", "cql_with_logprobs", "fqe", "bc_with_logprobs")


def _choose(name: str, value, options) -> str:
    options = list(options)
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


def health_score_group(hospital: str, cohort: str) -> str:
    """Folder of the RL models trained on this hospital's cohort, e.g. 'icu_patients_health_score_from_icu_dataset'."""
    _choose("hospital", hospital, HOSPITALS)
    return HOSPITALS[hospital]["cohorts"][_choose(f"cohort for {hospital}", cohort, HOSPITALS[hospital]["cohorts"])]


def _dynamic_deephit(models_root, hospital, cohort, action_inclusion, purpose, reward_type) -> str:
    if reward_type is not None:
        raise ValueError("reward_type only applies to RL models")
    if cohort is not None:
        raise ValueError("cohort is not used for dynamic_deephit (its folders are not split by cohort)")
    parts = [models_root, "dynamic_deephit_results", hospital, "split_0", "sepsis", HOSPITALS[hospital]["actions"],
             _choose("action_inclusion", action_inclusion, ACTION_INCLUSION)]
    if action_inclusion == "with_actions":
        parts.append(_choose("purpose (needed with actions)", purpose, DDH_PURPOSES))
    elif purpose is not None:
        raise ValueError("purpose only applies to dynamic_deephit trained with_actions")
    return os.path.join(*parts)


def _rl(model_type):
    def build(models_root, hospital, cohort, action_inclusion, purpose, reward_type):
        if purpose is not None:
            raise ValueError("purpose only applies to dynamic_deephit")
        # action_inclusion says which Dynamic-DeepHit variant supplies the RL states. Its position in the
        # tree is a guess (mirroring the DDH layout) until confirmed; folders below reward_type (discount,
        # sweep name) are not fixed here yet, so the path stops at the reward type.
        return os.path.join(models_root, "rl_models", hospital, RL_FOLDERS[model_type], "split_0", "sepsis",
                            HOSPITALS[hospital]["actions"], _choose("action_inclusion", action_inclusion, ACTION_INCLUSION),
                            _choose("reward_type (needed for RL models)", reward_type, REWARD_TYPES))
    return build


BUILDERS: Dict[str, Callable] = {"dynamic_deephit": _dynamic_deephit, **{m: _rl(m) for m in RL_MODEL_TYPES}}


def sweep_dir(models_root, model_type: str, hospital: str, cohort: Optional[str] = None,
              action_inclusion: str = "with_actions", purpose: Optional[str] = None,
              reward_type: Optional[str] = None) -> str:
    """
    Folder holding the runs of one sweep, checked to exist.
      model_type       : "dynamic_deephit", "basic_q", "cql_with_logprobs", "fqe" or "bc_with_logprobs"
      hospital         : "choc" or "mimiciv"
      cohort           : RL models only (see HOSPITALS[hospital]["cohorts"])
      action_inclusion : "with_actions" or "without_actions": the Dynamic-DeepHit variant (itself, or, for RL
                         models, the one supplying the states)
      purpose          : Dynamic-DeepHit with_actions only: "for_rewards" or "for_states"
      reward_type      : RL models only: "health_score_change_only", "terminal_only" or "terminal_bellman"
    """
    _choose("model_type", model_type, BUILDERS)
    hospital = HOSPITAL_ALIASES.get(hospital, hospital)
    _choose("hospital", hospital, HOSPITALS)
    path = BUILDERS[model_type](str(models_root), hospital, cohort, action_inclusion, purpose, reward_type)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"not found: {path}")
    return path