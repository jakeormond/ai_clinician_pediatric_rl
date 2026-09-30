"""
Vaso-reward positive control / confounding-by-indication sensitivity analysis.

Background
----------
The health-score CQL policy recommends zero vaso nearly everywhere. As a
positive control, models were retrained with an extra per-step reward
proportional to the clinician's (logged, shifted) vaso dose
(data_processor_for_shards.process_reward, `add_reward_to_vaso_dosing`):

    r_t = r_t^health  +  c * vis_{t+1} / max(vis)        (then * total_health_score_scale)

where c = hyperparams['vaso_reward_coefficient'] and vis_{t+1} is exactly the
shifted vaso action a_v(t) (actions_shift=True). On the TRAIN split
max(vis) == 1 (log_min_max normalisation fitted on train), so during training
the bonus is simply c * a_v in health-score points per hour. NOTE: on val/test
the divisor is that split's own max (<=1), so any val/test *rewards* are on a
slightly different scale -- irrelevant here (nothing below uses logged rewards
for the sweep models), but relevant if you ever run WIS/val-loss on these runs.

What this module measures, per coefficient c (and per seed)
-----------------------------------------------------------
Behaviour (what the actor does):
  - vaso fire rate (deterministic gate, p>0.5) overall / at clinician-vaso
    states / at clinician-zero states, and the continuous gate probability
    (far more sensitive than the hard threshold at low c)
  - AUROC of the gate probability for the clinician's vaso decision
    (does the policy learn WHEN vaso is given, or just fire everywhere?)
  - dose magnitudes, fluid side-effects (specificity of the manipulation)
  - SARSA-critic advantage (cost on the ORIGINAL, bonus-free objective)
    and occupancy support
Critic (why it does it):
  - dQ_vaso: Q(s, [f_clin, v_clin]) - Q(s, [f_clin, 0]) at clinician-vaso
    states, in health-score points (Q / total_health_score_scale), and the
    dose-weighted per-unit-dose version. Its zero crossing in c is the
    "tipping point" c*.
  - full Q(vaso dose) curves at clinician-vaso states.

Interpretation of c*
--------------------
c* is the size of per-hour, per-unit-VIS benefit that has to be ADDED to the
reward before the learned critic stops penalising vaso where clinicians gave
it. It is a tipping-point / E-value-style quantity: IF the c=0 penalty is
entirely due to confounding by indication, the unmeasured confounding is
worth ~c* health-score points per hour per unit (log-min-max-normalised) VIS.
The experiment does not by itself separate confounding from true harm or from
CQL/architecture effects; the data-level analyses (onset-aligned health score,
within-severity reward gaps) and the bonus-free SARSA reference are what argue
for the confounding interpretation.

Units caveat: "unit dose" is on the normalised (log1p, min-max) VIS scale.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from scipy import stats

from .evaluate_cql_alpha_sweep import (
    get_model_dirs,
    get_critic,
    compute_model_advantage,
    compute_occupancy_support_of_actions,
)
from src.rl.evaluate_terminal_reward_models import load_actor_and_critic


VASO_COEF_KEY = "vaso_reward_coefficient"
VASO_THRESHOLD = 1e-4          # same threshold used for "nonzero" elsewhere in the project

POLICY_COLOR = "#2471a3"
CLIN_COLOR = "#7f8c8d"
CLIN_VASO_COLOR = "#c0392b"
CLIN_ZERO_COLOR = "#2471a3"
SARSA_COLOR = "#7d3c98"
OCC_COLOR = "#229954"
FLUID_COLOR = "#d68910"
TIP_COLOR = "#7d3c98"


# =============================================================================
# Loading
# =============================================================================

def _checkpoint_hyperparams(model_dir):
    """Hyperparameters as saved in the run's own checkpoint (source of truth --
    config_manager.pkl has been found to be shared across sweeps)."""
    ckpt_path = os.path.join(model_dir, "final_model.pt")
    if not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("hyperparameters", None)


def load_vaso_sweep_models(sweep_dir, state_dim, baseline_dir=None, verify_checkpoints=True):
    """
    Loads every CQL run under `sweep_dir` and groups them by
    vaso_reward_coefficient. Returns {coef: [model_entry, ...]} where
    model_entry = (actor, critic, config_manager, hyperparams), same tuple
    shape as the alpha-sweep notebook.

    baseline_dir: optional folder of bonus-free CQL runs (e.g. the final
        fixed-alpha seeds) added as coef=0.0 if the sweep itself has no c=0
        run. They must share cql_alpha with the sweep for the comparison to
        be clean -- a warning is printed otherwise.

    verify_checkpoints: cross-checks coefficient / flag / cql_alpha / seed
        against each run's own final_model.pt and raises on mismatch.
    """
    models_by_coef = defaultdict(list)
    rows = []

    def _load_dir(model_dir, is_baseline):
        actor, critic, cm, hp = load_actor_and_critic(model_dir=model_dir, model_type="cql", state_dim=state_dim)
        flag = bool(hp.get("add_reward_to_vaso_dosing", False))
        coef = float(hp.get(VASO_COEF_KEY, 0.0)) if flag else 0.0

        if verify_checkpoints:
            ck = _checkpoint_hyperparams(model_dir)
            if ck is None:
                print(f"  WARNING: no final_model.pt hyperparameters in {model_dir}; cannot verify.")
            else:
                ck_flag = bool(ck.get("add_reward_to_vaso_dosing", False))
                ck_coef = float(ck.get(VASO_COEF_KEY, 0.0)) if ck_flag else 0.0
                for name, a, b in [("add_reward_to_vaso_dosing", flag, ck_flag),
                                   (VASO_COEF_KEY, coef, ck_coef),
                                   ("cql_alpha", hp.get("cql_alpha"), ck.get("cql_alpha")),
                                   ("seed", hp.get("seed"), ck.get("seed"))]:
                    if a != b and not (isinstance(a, float) and isinstance(b, float) and np.isclose(a, b)):
                        raise ValueError(f"{model_dir}: loaded {name}={a} but checkpoint says {b}. "
                                         f"Loaded hyperparams are not this run's own.")
        if is_baseline and flag:
            raise ValueError(f"Baseline run {model_dir} has add_reward_to_vaso_dosing=True.")
        if not is_baseline and not flag:
            print(f"  NOTE: sweep run {model_dir} has add_reward_to_vaso_dosing=False -> treated as c=0.")

        models_by_coef[coef].append((actor, critic, cm, hp))
        rows.append({"coef": coef, "seed": hp.get("seed"), "cql_alpha": hp.get("cql_alpha"),
                     "baseline": is_baseline, "dir": model_dir})

    for d in get_model_dirs(sweep_dir):
        _load_dir(d, is_baseline=False)

    if baseline_dir is not None:
        if 0.0 in models_by_coef:
            print("Sweep already contains c=0 runs; baseline runs are ADDED to c=0 as extra seeds.")
        for d in get_model_dirs(baseline_dir):
            _load_dir(d, is_baseline=True)

    summary = pd.DataFrame(rows).sort_values(["coef", "seed"]).reset_index(drop=True)
    alphas = summary["cql_alpha"].dropna().unique()
    if len(alphas) > 1:
        print(f"WARNING: runs use different cql_alpha values {alphas} -- c is confounded with alpha.")
    if len(models_by_coef) == 1:
        print("WARNING: only one coefficient found -- check the grouping key / shared config issue.")
    print(summary[["coef", "seed", "cql_alpha", "baseline"]].to_string(index=False))
    return dict(sorted(models_by_coef.items())), summary


# =============================================================================
# Per-state data helpers (health score, hourly change, severity)
# =============================================================================

def per_state_health_score(processor):
    """
    Health score and next-hour change, aligned 1:1 with processor.rl_dataset
    rows (each trajectory's final timestep dropped, as in the notebooks).
    Returns (hs, dhs) with dhs[i] = HS(t+1) - HS(t): the unclipped,
    unscaled health component of the reward at state i.
    """
    hs_full = np.asarray(processor.dynamic_deephit_predictions["health_score"], dtype=np.float64)
    ptr = np.asarray(processor.dynamic_deephit_predictions["index_ptr"])
    hs, dhs = [], []
    for s, e in zip(ptr[:-1], ptr[1:]):
        traj = hs_full[s:e]
        hs.append(traj[:-1])
        dhs.append(np.diff(traj))
    hs, dhs = np.concatenate(hs), np.concatenate(dhs)
    n_states = processor.rl_dataset["states"].shape[0]
    if len(hs) != n_states:
        raise ValueError(f"health score alignment failed: {len(hs)} vs {n_states} RL states")
    return hs, dhs


def severity_edges_from_train(train_hs, n_bins=10):
    """Quantile edges from TRAIN health score (low health score = sicker)."""
    edges = np.quantile(train_hs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def severity_bin(hs, edges):
    return np.clip(np.digitize(hs, edges[1:-1]), 0, len(edges) - 2)


def trajectory_ids(processor):
    """Integer trajectory id for each RL row, from the dones vector."""
    dones = np.asarray(processor.rl_dataset["dones"]).astype(bool)
    return np.concatenate(([0], np.cumsum(dones)[:-1]))


# =============================================================================
# Model inference (torch)
# =============================================================================

def _device_of(module):
    return next(module.parameters()).device


@torch.no_grad()
def actor_gate_outputs(actor, states, batch_size=4096):
    """
    Deterministic actor outputs for every state. Actions are recomputed
    exactly as Actor.forward(deterministic=True) does, so they should match
    generate_model_outputs' actions (checked in summarise_model).
    """
    actor.eval()
    dev = _device_of(actor)
    out = defaultdict(list)
    for i in range(0, states.shape[0], batch_size):
        s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=dev)
        actions, _, _, dist = actor(s, deterministic=True, return_dist=True)
        f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std = dist
        out["actions"].append(actions.cpu())
        out["f_p"].append(f_p.squeeze(-1).cpu())
        out["v_p"].append(v_p.squeeze(-1).cpu())
        out["v_mag"].append(torch.sigmoid(v_raw_mean).squeeze(-1).cpu())
        out["v_std"].append(v_std.squeeze(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def critic_q(critic, states, actions, batch_size=4096):
    """Mean of the twin Q heads (matches reduction='mean' elsewhere)."""
    critic.eval()
    dev = _device_of(critic)
    qs = []
    for i in range(0, states.shape[0], batch_size):
        s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=dev)
        a = torch.as_tensor(actions[i:i + batch_size], dtype=torch.float32, device=dev)
        q1, q2 = critic(s, a)
        qs.append(((q1 + q2) / 2).squeeze(-1).cpu())
    return torch.cat(qs).numpy()


def q_vaso_dose_curves(critic, states, clin_actions, vaso_grid, batch_size=4096):
    """
    Q(s, [f_clin, v]) for v in vaso_grid, holding the clinician's fluid fixed.
    Returns (n_states, n_grid).
    """
    clin_actions = np.asarray(clin_actions, dtype=np.float32)
    out = np.zeros((len(states), len(vaso_grid)), dtype=np.float64)
    for j, v in enumerate(vaso_grid):
        a = clin_actions.copy()
        a[:, 1] = v
        out[:, j] = critic_q(critic, states, a, batch_size=batch_size)
    return out


def vaso_delta_q(critic, states, clin_actions, batch_size=4096):
    """Q(s, [f_clin, v_clin]) - Q(s, [f_clin, 0]) per state (raw Q units)."""
    clin_actions = np.asarray(clin_actions, dtype=np.float32)
    zero_v = clin_actions.copy()
    zero_v[:, 1] = 0.0
    return critic_q(critic, states, clin_actions, batch_size) - critic_q(critic, states, zero_v, batch_size)


def _reward_scale(hp, name="model"):
    """Divisor that converts Q units back to health-score points."""
    scale = hp.get("total_health_score_scale", None) if isinstance(hp, dict) else None
    if scale is None:
        print(f"  WARNING: {name} has no total_health_score_scale; dQ left in raw Q units.")
        return 1.0
    hc = hp.get("health_score_hourly_change_scale", 1.0)
    if hc != 1.0:
        print(f"  NOTE: {name} health_score_hourly_change_scale={hc}; health units are scaled points.")
    return float(scale)


# =============================================================================
# Core per-model summary
# =============================================================================

def summarise_model(model_entry, processor, hs, sev_edges, sarsa_models,
                    training_actions, training_health_score,
                    vaso_grid=np.linspace(0, 1, 11), n_severity_bins_occ=200,
                    min_count=10, compute_sarsa=True):
    actor, critic, _, hp = model_entry
    states = processor.rl_dataset["states"]
    states_np = states.numpy() if torch.is_tensor(states) else np.asarray(states)
    clin = processor.rl_dataset["actions"]
    clin = clin.numpy() if torch.is_tensor(clin) else np.asarray(clin)

    clin_v_fire = clin[:, 1] > VASO_THRESHOLD
    clin_f_fire = clin[:, 0] > VASO_THRESHOLD
    sev = severity_bin(hs, sev_edges)

    g = actor_gate_outputs(actor, states_np)
    pol = g["actions"]
    pol_v_fire = pol[:, 1] > VASO_THRESHOLD
    pol_f_fire = pol[:, 0] > VASO_THRESHOLD

    r = {"seed": hp.get("seed"), "cql_alpha": hp.get("cql_alpha")}

    # --- behaviour ---
    r["vaso_fire_rate"] = pol_v_fire.mean()
    r["vaso_fire_rate_at_clin_vaso"] = pol_v_fire[clin_v_fire].mean()
    r["vaso_fire_rate_at_clin_zero"] = pol_v_fire[~clin_v_fire].mean()
    r["vaso_gate_p_mean"] = g["v_p"].mean()
    r["vaso_gate_p_at_clin_vaso"] = g["v_p"][clin_v_fire].mean()
    r["vaso_gate_p_at_clin_zero"] = g["v_p"][~clin_v_fire].mean()
    r["vaso_gate_auroc"] = roc_auc_score(clin_v_fire, g["v_p"]) if 0 < clin_v_fire.mean() < 1 else np.nan
    r["vaso_dose_mean"] = pol[:, 1].mean()
    r["vaso_dose_mean_when_fired"] = pol[pol_v_fire, 1].mean() if pol_v_fire.any() else np.nan
    r["vaso_magnitude_mean"] = g["v_mag"].mean()      # dose it WOULD give if gate opened
    both = pol_v_fire & clin_v_fire
    r["vaso_dose_spearman_cofired"] = (stats.spearmanr(pol[both, 1], clin[both, 1])[0]
                                       if both.sum() > 10 else np.nan)
    r["fluid_fire_rate"] = pol_f_fire.mean()
    r["fluid_dose_mean"] = pol[:, 0].mean()

    r["vaso_fire_rate_by_severity"] = np.array(
        [pol_v_fire[sev == b].mean() if (sev == b).any() else np.nan for b in range(len(sev_edges) - 1)])
    r["vaso_gate_p_by_severity"] = np.array(
        [g["v_p"][sev == b].mean() if (sev == b).any() else np.nan for b in range(len(sev_edges) - 1)])

    # --- own critic: vaso penalty/bonus at clinician-vaso states ---
    scale = _reward_scale(hp, "CQL")
    idx = np.where(clin_v_fire)[0]
    dq = vaso_delta_q(critic, states_np[idx], clin[idx]) / scale
    r["dq_vaso_mean"] = dq.mean()
    r["dq_vaso_per_unit_dose"] = dq.sum() / clin[idx, 1].sum()   # dose-weighted, stable
    r["dq_vaso_frac_positive"] = (dq > 0).mean()
    r["dq_vaso_states"] = dq
    r["dq_vaso_severity"] = sev[idx]
    curves = q_vaso_dose_curves(critic, states_np[idx], clin[idx], vaso_grid) / scale
    r["q_vaso_curve_mean"] = (curves - curves[:, [0]]).mean(axis=0)
    r["vaso_grid"] = np.asarray(vaso_grid)

    # --- independent (bonus-free) SARSA judges + occupancy ---
    if compute_sarsa and sarsa_models:
        adv = compute_model_advantage(model_entry, processor, sarsa_models)
        a_ref = adv["actor_actions"]
        a_ref = a_ref.cpu().numpy() if torch.is_tensor(a_ref) else np.asarray(a_ref)
        max_diff = np.abs(a_ref - pol).max()
        if max_diff > 1e-4:
            print(f"  WARNING: actor_gate_outputs actions differ from generate_model_outputs by {max_diff:.2e}")
        r["sarsa_adv_norm_by_seed"] = np.array([a.mean() for a in adv["sarsa_advantages_normalized"]])
        r["sarsa_adv_norm"] = r["sarsa_adv_norm_by_seed"].mean()
        # own-critic advantage is NOT comparable across c (reward differs) -- kept for completeness
        r["own_adv_norm"] = adv["own_advantage_normalized"].mean()

    occ_frac, _ = compute_occupancy_support_of_actions(
        pol, hs, training_actions, training_health_score,
        n_severity_bins=n_severity_bins_occ, min_count=min_count)
    r["occupancy_support_frac"] = occ_frac
    return r


def clinician_reference(processor, hs, sev_edges):
    clin = processor.rl_dataset["actions"]
    clin = clin.numpy() if torch.is_tensor(clin) else np.asarray(clin)
    v_fire = clin[:, 1] > VASO_THRESHOLD
    sev = severity_bin(hs, sev_edges)
    return {
        "vaso_fire_rate": v_fire.mean(),
        "vaso_dose_mean": clin[:, 1].mean(),
        "vaso_dose_mean_when_fired": clin[v_fire, 1].mean(),
        "fluid_fire_rate": (clin[:, 0] > VASO_THRESHOLD).mean(),
        "fluid_dose_mean": clin[:, 0].mean(),
        "vaso_fire_rate_by_severity": np.array(
            [v_fire[sev == b].mean() for b in range(len(sev_edges) - 1)]),
    }


def sarsa_vaso_reference(processor, sarsa_models, vaso_grid=np.linspace(0, 1, 11)):
    """
    Bonus-free, non-conservative reference: the same dQ_vaso measures on each
    SARSA critic. If these are also negative, the zero-vaso preference is not
    a CQL-conservatism artefact -- it is in the logged data's (s, a, r) relation.
    """
    states = processor.rl_dataset["states"]
    states = states.numpy() if torch.is_tensor(states) else np.asarray(states)
    clin = processor.rl_dataset["actions"]
    clin = clin.numpy() if torch.is_tensor(clin) else np.asarray(clin)
    idx = np.where(clin[:, 1] > VASO_THRESHOLD)[0]
    out = []
    for k, entry in enumerate(sarsa_models):
        critic = get_critic(entry)
        hp = entry[2] if isinstance(entry, (list, tuple)) and len(entry) >= 3 else {}
        scale = _reward_scale(hp, f"SARSA seed {k + 1}")
        dq = vaso_delta_q(critic, states[idx], clin[idx]) / scale
        curves = q_vaso_dose_curves(critic, states[idx], clin[idx], vaso_grid) / scale
        out.append({"dq_vaso_mean": dq.mean(),
                    "dq_vaso_per_unit_dose": dq.sum() / clin[idx, 1].sum(),
                    "dq_vaso_frac_positive": (dq > 0).mean(),
                    "dq_vaso_states": dq,
                    "q_vaso_curve_mean": (curves - curves[:, [0]]).mean(axis=0),
                    "vaso_grid": np.asarray(vaso_grid)})
    return out


def run_vaso_reward_sweep_analysis(models_by_coef, processor, sarsa_models, hs, sev_edges,
                                    training_actions, training_health_score, **kwargs):
    """Returns {coef: [per-seed result dict, ...]}."""
    results = {}
    for coef, entries in models_by_coef.items():
        results[coef] = []
        for entry in entries:
            t0 = time.time()
            print(f"=== vaso_reward_coefficient={coef}, seed={entry[3].get('seed')} ===", flush=True)
            r = summarise_model(entry, processor, hs, sev_edges, sarsa_models,
                                training_actions, training_health_score, **kwargs)
            print(f"  vaso fire: {r['vaso_fire_rate']*100:.1f}% overall, "
                  f"{r['vaso_fire_rate_at_clin_vaso']*100:.1f}% @clin-vaso, "
                  f"{r['vaso_fire_rate_at_clin_zero']*100:.1f}% @clin-zero | "
                  f"gate AUROC {r['vaso_gate_auroc']:.3f} | "
                  f"dQ/unit dose {r['dq_vaso_per_unit_dose']:+.3f} HS pts | "
                  f"({time.time() - t0:.0f}s)", flush=True)
            results[coef].append(r)
    return results


# =============================================================================
# Aggregation, tables, tipping points
# =============================================================================

SCALAR_KEYS = [
    "vaso_fire_rate", "vaso_fire_rate_at_clin_vaso", "vaso_fire_rate_at_clin_zero",
    "vaso_gate_p_mean", "vaso_gate_p_at_clin_vaso", "vaso_gate_p_at_clin_zero", "vaso_gate_auroc",
    "vaso_dose_mean", "vaso_dose_mean_when_fired", "vaso_magnitude_mean", "vaso_dose_spearman_cofired",
    "fluid_fire_rate", "fluid_dose_mean",
    "dq_vaso_mean", "dq_vaso_per_unit_dose", "dq_vaso_frac_positive",
    "sarsa_adv_norm", "own_adv_norm", "occupancy_support_frac",
]


def results_to_dataframe(results, per_seed=False):
    rows = []
    for coef, seeds in results.items():
        for r in seeds:
            rows.append({"coef": coef, "seed": r.get("seed"),
                         **{k: r.get(k, np.nan) for k in SCALAR_KEYS}})
    df = pd.DataFrame(rows)
    if per_seed:
        return df
    agg = df.drop(columns="seed").groupby("coef").agg(["mean", "min", "max"])
    agg.insert(0, ("n_seeds", ""), df.groupby("coef").size())
    return agg


def _agg(results, key):
    coefs = sorted(results)
    vals = [np.array([r.get(key, np.nan) for r in results[c]], dtype=float) for c in coefs]
    return (np.array(coefs), np.array([np.nanmean(v) for v in vals]),
            np.array([np.nanmin(v) for v in vals]), np.array([np.nanmax(v) for v in vals]), vals)


def _crossing(x, y, level=0.0):
    """First x at which y crosses `level` from below (linear interpolation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            return x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    return np.nan if y[0] < level else x[0]


def estimate_tipping_points(results, clin_ref):
    """
    c* estimates (on the seed-mean curves, and per seed where seeds are
    shared across coefficients):
      critic:     dose-weighted dQ_vaso per unit dose crosses 0
      behaviour:  fire rate at clinician-vaso states crosses 50%
      rate-match: overall fire rate reaches the clinician's overall rate
    NaN = never crossed within the swept range.
    """
    defs = {"critic (dQ/unit dose = 0)": ("dq_vaso_per_unit_dose", 0.0),
            "behaviour (fires at 50% of clin-vaso states)": ("vaso_fire_rate_at_clin_vaso", 0.5),
            "rate-match (overall fire rate = clinician)": ("vaso_fire_rate", clin_ref["vaso_fire_rate"])}
    out = {}
    for name, (key, level) in defs.items():
        coefs, mean, *_ = _agg(results, key)
        per_seed = {}
        seeds = [{r.get("seed") for r in results[c]} for c in coefs]
        shared = set.intersection(*seeds) if seeds else set()
        for sd in sorted(s for s in shared if s is not None):
            y = [np.mean([r[key] for r in results[c] if r.get("seed") == sd]) for c in coefs]
            per_seed[sd] = _crossing(coefs, y, level)
        out[name] = {"c_star": _crossing(coefs, mean, level), "per_seed": per_seed}
    for name, v in out.items():
        ps = ", ".join(f"s{k}:{c:.3g}" for k, c in v["per_seed"].items())
        print(f"{name:48s} c* = {v['c_star']:.4g}" + (f"   [{ps}]" if ps else ""))
    return out


def contextualise_tipping_point(c_star, train_dhs, hp):
    """
    Puts c* on interpretable scales:
      - relative to the spread of real hourly health-score change
      - relative to the reward clip used in training
      - as a discounted-horizon equivalent: c*/(1-gamma) = total value of the
        bonus if a unit dose were sustained indefinitely
    """
    clip = hp.get("health_score_hourly_change_clip", np.nan)
    gamma = hp.get("discount", np.nan)
    sd = np.std(train_dhs)
    out = {
        "c_star (HS pts/hour per unit VIS)": c_star,
        "c_star / SD(hourly HS change)": c_star / sd,
        "percentile of |hourly HS change|": stats.percentileofscore(np.abs(train_dhs), c_star),
        "c_star / hourly clip": c_star / clip if clip else np.nan,
        "sustained-dose value c*/(1-gamma) (HS pts)": c_star / (1 - gamma) if gamma == gamma else np.nan,
    }
    for k, v in out.items():
        print(f"  {k:45s} {v:.3g}")
    return out


# =============================================================================
# Data-level confounding-by-indication analyses (no models)
# =============================================================================

def vaso_onset_aligned(processor, hs, dhs, sev_edges, pre=12, post=12, washout=6,
                       n_controls=5, seed=0):
    """
    Health score and hourly change aligned to clinician vaso INITIATION
    (first nonzero vaso after >= `washout` vaso-free hours in the same stay),
    plus severity-matched controls: states in the same severity bin with no
    vaso anywhere in [-pre, +post] around them.
    Confounding-by-indication signature: vaso patients are already
    deteriorating before onset despite matched current severity, and their
    next-hour change (the reward credited to the vaso action) is negative.
    """
    clin_v = np.asarray(processor.rl_dataset["actions"][:, 1]) > VASO_THRESHOLD
    tid = trajectory_ids(processor)
    sev = severity_bin(hs, sev_edges)
    n = len(hs)
    lags = np.arange(-pre, post + 1)

    def window(i, arr):
        w = np.full(len(lags), np.nan)
        for j, k in enumerate(lags):
            t = i + k
            if 0 <= t < n and tid[t] == tid[i]:
                w[j] = arr[t]
        return w

    onsets = []
    for i in np.where(clin_v)[0]:
        lo = i - washout
        if lo >= 0 and tid[lo] == tid[i] and not clin_v[lo:i].any():
            onsets.append(i)
    onsets = np.array(onsets, dtype=int)

    # control pool: no vaso within the window, same trajectory-bounded window
    vaso_any = np.zeros(n, bool)
    for i in np.where(clin_v)[0]:
        vaso_any[max(0, i - post):min(n, i + pre + 1)] = True
    pool = np.where(~vaso_any)[0]
    rng = np.random.default_rng(seed)
    controls = []
    for i in onsets:
        cand = pool[sev[pool] == sev[i]]
        if len(cand):
            controls.extend(rng.choice(cand, size=min(n_controls, len(cand)), replace=False))
    controls = np.array(controls, dtype=int)

    return {
        "lags": lags, "n_onsets": len(onsets), "n_controls": len(controls),
        "onset_hs": np.array([window(i, hs) for i in onsets]),
        "onset_dhs": np.array([window(i, dhs) for i in onsets]),
        "control_hs": np.array([window(i, hs) for i in controls]),
        "control_dhs": np.array([window(i, dhs) for i in controls]),
    }


def reward_gap_by_severity(processor, hs, dhs, sev_edges):
    """Mean next-hour HS change at clinician-vaso vs no-vaso states, per severity bin."""
    clin_v = np.asarray(processor.rl_dataset["actions"][:, 1]) > VASO_THRESHOLD
    sev = severity_bin(hs, sev_edges)
    rows = []
    for b in range(len(sev_edges) - 1):
        m = sev == b
        v, z = dhs[m & clin_v], dhs[m & ~clin_v]
        rows.append({"severity_bin": b, "n_vaso": len(v), "n_zero": len(z),
                     "dhs_vaso": v.mean() if len(v) else np.nan,
                     "dhs_zero": z.mean() if len(z) else np.nan,
                     "gap": (v.mean() - z.mean()) if len(v) and len(z) else np.nan,
                     "gap_sem": np.sqrt(v.var() / len(v) + z.var() / len(z)) if len(v) > 1 and len(z) > 1 else np.nan})
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def _set_coef_xscale(ax, coefs, log=None):
    coefs = np.asarray(coefs, float)
    nz = coefs[coefs > 0]
    if log is None:
        log = len(nz) > 1 and nz.max() / nz.min() >= 20
    if log:
        if (coefs == 0).any():
            ax.set_xscale("symlog", linthresh=nz.min())
        else:
            ax.set_xscale("log")
    ax.set_xlabel("vaso_reward_coefficient  c")


def _line(ax, results, key, color, label, scale=1.0, marker="o", ls="-"):
    coefs, mean, lo, hi, vals = _agg(results, key)
    ax.plot(coefs, mean * scale, marker=marker, color=color, label=label, ls=ls)
    if any(len(v) > 1 for v in vals):
        ax.fill_between(coefs, lo * scale, hi * scale, color=color, alpha=0.15)
    return coefs


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot_vaso_dosing_vs_coef(results, clin_ref, log_x=None, title="", save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    coefs = _line(ax, results, "vaso_fire_rate", "k", "all states", 100)
    _line(ax, results, "vaso_fire_rate_at_clin_vaso", CLIN_VASO_COLOR, "clinician gave vaso", 100)
    _line(ax, results, "vaso_fire_rate_at_clin_zero", CLIN_ZERO_COLOR, "clinician gave none", 100, ls="--")
    ax.axhline(clin_ref["vaso_fire_rate"] * 100, color=CLIN_COLOR, ls=":", label="clinician rate")
    ax.set_ylabel("% states with vaso (deterministic)")
    ax.set_title("Vaso fire rate", fontsize=10)

    ax = axes[0, 1]
    _line(ax, results, "vaso_gate_p_at_clin_vaso", CLIN_VASO_COLOR, "clinician gave vaso")
    _line(ax, results, "vaso_gate_p_at_clin_zero", CLIN_ZERO_COLOR, "clinician gave none", ls="--")
    ax.axhline(0.5, color=CLIN_COLOR, ls=":", lw=0.8)
    ax.set_ylabel("mean gate probability P(vaso>0)")
    ax.set_title("Vaso gate probability (continuous)", fontsize=10)

    ax = axes[0, 2]
    _line(ax, results, "vaso_gate_auroc", POLICY_COLOR, "gate AUROC")
    ax.axhline(0.5, color=CLIN_COLOR, ls=":", lw=0.8)
    ax.set_ylabel("AUROC vs clinician vaso decision")
    ax.set_title("Does the gate track WHEN clinicians give vaso?", fontsize=10)

    ax = axes[1, 0]
    _line(ax, results, "vaso_dose_mean_when_fired", POLICY_COLOR, "policy, when fired")
    _line(ax, results, "vaso_magnitude_mean", POLICY_COLOR, "policy magnitude head (ungated)", ls="--", marker="s")
    ax.axhline(clin_ref["vaso_dose_mean_when_fired"], color=CLIN_COLOR, ls=":", label="clinician, when fired")
    ax.set_ylabel("normalised vaso dose")
    ax.set_title("Vaso dose magnitude", fontsize=10)

    ax = axes[1, 1]
    _line(ax, results, "fluid_fire_rate", FLUID_COLOR, "fluid fire rate (%)", 100)
    ax.axhline(clin_ref["fluid_fire_rate"] * 100, color=FLUID_COLOR, ls=":", label="clinician fluid rate")
    ax.set_ylabel("% states with fluid")
    axb = ax.twinx()
    _line(axb, results, "fluid_dose_mean", "k", "mean fluid dose", marker="s", ls="--")
    axb.set_ylabel("mean fluid dose")
    ax.set_title("Off-target check: fluid", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)

    ax = axes[1, 2]
    if all("sarsa_adv_norm" in r for s in results.values() for r in s):
        _line(ax, results, "sarsa_adv_norm", SARSA_COLOR, "SARSA advantage (bonus-free)")
        ax.axhline(0, color=CLIN_COLOR, ls=":", lw=0.8)
        ax.set_ylabel("SARSA advantage (frac. of Q IQR)", color=SARSA_COLOR)
    axo = ax.twinx()
    _line(axo, results, "occupancy_support_frac", OCC_COLOR, "occupancy support (%)", 100, marker="s", ls="--")
    axo.set_ylabel("% actions well-supported", color=OCC_COLOR)
    ax.set_title("Cost on original objective & data support", fontsize=10)

    for ax in axes.flat:
        _set_coef_xscale(ax, coefs, log_x)
        _style(ax)
        if ax is not axes[1, 1]:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title or "Vaso-reward positive control: policy vaso dosing vs. bonus coefficient",
                 fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_critic_tipping_point(results, sarsa_ref=None, tipping=None, log_x=None, save_path=None):
    """
    Left: dose-weighted dQ_vaso per unit dose (HS points) vs c, with the
    bonus-free SARSA critics' value as a reference band and c* marked.
    Right: the behavioural response (gate prob at clinician-vaso states)
    on the same x axis, so critic and actor tipping points can be compared.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    coefs = _line(ax, results, "dq_vaso_per_unit_dose", POLICY_COLOR, "CQL own critic")
    ax.axhline(0, color=CLIN_COLOR, lw=0.8, ls="--")
    if sarsa_ref:
        v = [s["dq_vaso_per_unit_dose"] for s in sarsa_ref]
        ax.axhspan(min(v), max(v), color=SARSA_COLOR, alpha=0.15, label="SARSA (c=0) range")
        ax.axhline(np.mean(v), color=SARSA_COLOR, lw=1, label="SARSA (c=0) mean")
    ax.set_ylabel("dQ per unit vaso dose (health-score points)")
    ax.set_title("Critic: value of clinician's vaso vs. none\n(at clinician-vaso states)", fontsize=10)

    ax2 = axes[1]
    _line(ax2, results, "vaso_gate_p_at_clin_vaso", CLIN_VASO_COLOR, "gate P at clinician-vaso states")
    _line(ax2, results, "vaso_fire_rate_at_clin_vaso", CLIN_VASO_COLOR, "fire rate at clinician-vaso states",
          marker="s", ls="--")
    ax2.axhline(0.5, color=CLIN_COLOR, lw=0.8, ls="--")
    ax2.set_ylabel("probability / fraction")
    ax2.set_title("Actor response", fontsize=10)

    if tipping:
        styles = {"critic": "-", "behaviour": "--", "rate-match": ":"}
        for name, v in tipping.items():
            c = v["c_star"]
            if np.isfinite(c):
                ls = styles.get(name.split(" ")[0], "-.")
                for a in (ax, ax2):
                    a.axvline(c, color=TIP_COLOR, ls=ls, lw=1.2,
                              label=f"c* {name.split(' ')[0]} = {c:.3g}" if a is ax else None)
    for a in (ax, ax2):
        _set_coef_xscale(a, coefs, log_x)
        _style(a)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_q_vaso_dose_curves(results, sarsa_ref=None, coefs_to_show=None, save_path=None):
    """Mean Q(vaso dose) - Q(0) at clinician-vaso states, one line per c (seed-averaged)."""
    coefs = sorted(results) if coefs_to_show is None else coefs_to_show
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(coefs):
        curves = np.array([r["q_vaso_curve_mean"] for r in results[c]])
        grid = results[c][0]["vaso_grid"]
        ax.plot(grid, curves.mean(0), color=cmap(i / max(1, len(coefs) - 1)), marker="o", ms=3, label=f"c={c:g}")
    if sarsa_ref:
        sc = np.array([s["q_vaso_curve_mean"] for s in sarsa_ref])
        ax.plot(sarsa_ref[0]["vaso_grid"], sc.mean(0), color="k", ls="--", lw=2, label="SARSA (c=0)")
    ax.axhline(0, color=CLIN_COLOR, lw=0.8)
    ax.set_xlabel("vaso dose (normalised), fluid held at clinician's")
    ax.set_ylabel("Q(v) - Q(0)  (health-score points)")
    ax.set_title("Critic vaso dose-response at clinician-vaso states", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_vaso_by_severity(results, clin_ref, key="vaso_fire_rate_by_severity", save_path=None):
    """Heatmap: rows = c (seed-mean), cols = severity decile; clinician row on top."""
    coefs = sorted(results)
    mat = np.array([np.nanmean([r[key] for r in results[c]], axis=0) for c in coefs])
    clin_row = clin_ref["vaso_fire_rate_by_severity"][None, :]
    full = np.vstack([clin_row, mat]) * 100
    labels = ["clinician"] + [f"c={c:g}" for c in coefs]
    fig, ax = plt.subplots(figsize=(1 + 0.7 * full.shape[1], 1 + 0.4 * full.shape[0]))
    im = ax.imshow(full, aspect="auto", cmap="magma_r", vmin=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(full.shape[1]), [f"D{i+1}" for i in range(full.shape[1])])
    ax.set_xlabel("health-score decile (D1 = sickest)")
    ax.axhline(0.5, color="w", lw=2)
    for (i, j), v in np.ndenumerate(full):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                color="w" if v > np.nanmax(full) / 2 else "k")
    fig.colorbar(im, ax=ax, label="% states" if "rate" in key else "mean gate P x100")
    ax.set_title("Vaso by severity: does the policy's vaso follow the clinicians' indication?", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_vaso_penalty_by_severity(results, sev_edges, sarsa_ref=None, coefs_to_show=None,
                                  sarsa_sev=None, save_path=None):
    """
    Per-severity-decile mean dQ_vaso (clinician dose vs none) at clinician-vaso
    states. At c=0, a penalty that is largest in the sickest deciles is the
    pattern expected under confounding by indication.
    sarsa_sev: severity bins for the clinician-vaso states (same ordering as
    the SARSA dq arrays), needed to add the SARSA line.
    """
    n_bins = len(sev_edges) - 1
    coefs = sorted(results) if coefs_to_show is None else coefs_to_show
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    def by_bin(dq, sev):
        return np.array([dq[sev == b].mean() if (sev == b).sum() >= 5 else np.nan for b in range(n_bins)])

    for i, c in enumerate(coefs):
        m = np.nanmean([by_bin(r["dq_vaso_states"], r["dq_vaso_severity"]) for r in results[c]], axis=0)
        ax.plot(np.arange(1, n_bins + 1), m, marker="o", color=cmap(i / max(1, len(coefs) - 1)), label=f"c={c:g}")
    if sarsa_ref and sarsa_sev is not None:
        m = np.nanmean([by_bin(s["dq_vaso_states"], sarsa_sev) for s in sarsa_ref], axis=0)
        ax.plot(np.arange(1, n_bins + 1), m, color="k", ls="--", lw=2, label="SARSA (c=0)")
    ax.axhline(0, color=CLIN_COLOR, lw=0.8)
    ax.set_xlabel("health-score decile (1 = sickest)")
    ax.set_ylabel("dQ: clinician vaso vs none (HS points)")
    ax.set_title("Critic's vaso penalty by severity (clinician-vaso states)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_vaso_onset(onset, save_path=None):
    lags = onset["lags"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, ylabel in [(axes[0], "hs", "health score"),
                            (axes[1], "dhs", "next-hour health-score change\n(= health reward at that state)")]:
        for grp, color, lab in [("onset", CLIN_VASO_COLOR, f"vaso initiation (n={onset['n_onsets']})"),
                                ("control", CLIN_COLOR, f"severity-matched controls (n={onset['n_controls']})")]:
            arr = onset[f"{grp}_{key}"]
            if arr.size == 0:
                continue
            m = np.nanmean(arr, 0)
            se = np.nanstd(arr, 0) / np.sqrt(np.sum(~np.isnan(arr), 0).clip(1))
            ax.plot(lags, m, color=color, label=lab)
            ax.fill_between(lags, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.2)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        if key == "dhs":
            ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("hours relative to vaso initiation (t=0: first vaso action)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    axes[0].set_title("Health score around vaso initiation", fontsize=10)
    axes[1].set_title("Reward credited around vaso initiation", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_reward_gap_by_severity(gap_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    x = gap_df["severity_bin"] + 1
    axes[0].plot(x, gap_df["dhs_vaso"], marker="o", color=CLIN_VASO_COLOR, label="clinician gave vaso")
    axes[0].plot(x, gap_df["dhs_zero"], marker="o", color=CLIN_ZERO_COLOR, label="clinician gave none")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("mean next-hour HS change")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(x, gap_df["gap"], yerr=1.96 * gap_df["gap_sem"], marker="o", color="k", capsize=3)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("vaso - no-vaso (HS points/hour, 95% CI)")
    for ax in axes:
        ax.set_xlabel("health-score decile (1 = sickest)")
        _style(ax)
    axes[0].set_title("Immediate health reward by severity", fontsize=10)
    axes[1].set_title("Within-severity reward gap attributed to vaso", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# Numerical summaries for sections 4-6 (tables mirroring the figures)
# =============================================================================

def behaviour_table(results, clin_ref):
    """Section 4 numbers: seed-mean behaviour per c, plus a clinician row.
    PPV = fraction of the policy's vaso recommendations made at states where
    the clinician also gave vaso (derived from the stored rates)."""
    prev = clin_ref["vaso_fire_rate"]
    rows = []
    for c in sorted(results):
        m = lambda k: np.nanmean([r.get(k, np.nan) for r in results[c]])
        fr, fr_v = m("vaso_fire_rate"), m("vaso_fire_rate_at_clin_vaso")
        rows.append({"coef": c, "n_seeds": len(results[c]),
                     "fire_%": 100 * fr, "fire_%_clin_vaso": 100 * fr_v,
                     "fire_%_clin_zero": 100 * m("vaso_fire_rate_at_clin_zero"),
                     "PPV_%": 100 * fr_v * prev / fr if fr > 0 else np.nan,
                     "gate_p_mean": m("vaso_gate_p_mean"), "gate_auroc": m("vaso_gate_auroc"),
                     "dose_when_fired": m("vaso_dose_mean_when_fired"),
                     "magnitude_mean": m("vaso_magnitude_mean"),
                     "dose_spearman_cofired": m("vaso_dose_spearman_cofired"),
                     "fluid_fire_%": 100 * m("fluid_fire_rate"), "fluid_dose_mean": m("fluid_dose_mean"),
                     "sarsa_adv_norm": m("sarsa_adv_norm"),
                     "occupancy_%": 100 * m("occupancy_support_frac")})
    df = pd.DataFrame(rows).set_index("coef")
    clin = pd.DataFrame([{"fire_%": 100 * prev, "fire_%_clin_vaso": 100.0, "fire_%_clin_zero": 0.0,
                          "PPV_%": 100.0, "dose_when_fired": clin_ref["vaso_dose_mean_when_fired"],
                          "fluid_fire_%": 100 * clin_ref["fluid_fire_rate"],
                          "fluid_dose_mean": clin_ref["fluid_dose_mean"]}], index=["clinician"])
    return pd.concat([clin, df])


def critic_table(results, sarsa_ref=None):
    """Section 5 numbers: dQ per unit dose, fraction of states favouring the
    clinician's vaso, and the local slope d(dQ)/dc between consecutive c
    (~1 expected if the gap carries only the immediate bonus)."""
    coefs, dq, *_ = _agg(results, "dq_vaso_per_unit_dose")
    _, frac, *_ = _agg(results, "dq_vaso_frac_positive")
    _, dq_raw, *_ = _agg(results, "dq_vaso_mean")
    slope = np.r_[np.nan, np.diff(dq) / np.diff(coefs)]
    df = pd.DataFrame({"dq_per_unit_dose": dq, "dq_at_clin_dose": dq_raw,
                       "frac_states_dq>0_%": 100 * frac, "slope_vs_prev_c": slope},
                      index=pd.Index(coefs, name="coef"))
    if sarsa_ref:
        for k, s in enumerate(sarsa_ref):
            df.loc[f"SARSA_{k + 1}"] = [s["dq_vaso_per_unit_dose"], s["dq_vaso_mean"],
                                        100 * s["dq_vaso_frac_positive"], np.nan]
    return df


def q_curve_table(results, sarsa_ref=None):
    """Mean Q(v) - Q(0) at clinician-vaso states (HS points); rows c, cols dose.
    argmax column = critic's preferred dose on average."""
    grid = next(iter(results.values()))[0]["vaso_grid"]
    rows = {c: np.nanmean([r["q_vaso_curve_mean"] for r in results[c]], axis=0) for c in sorted(results)}
    if sarsa_ref:
        rows["SARSA_mean"] = np.mean([s["q_vaso_curve_mean"] for s in sarsa_ref], axis=0)
    df = pd.DataFrame(rows, index=[f"v={v:.1f}" for v in grid]).T
    df["argmax_dose"] = grid[np.argmax(df[[f"v={v:.1f}" for v in grid]].values, axis=1)]
    return df


def severity_tables(results, clin_ref, sev_edges, sarsa_ref=None, sarsa_sev=None):
    """Section 6 numbers: (a) vaso fire rate (%) by severity decile, clinician
    row first; (b) critic dQ (clinician vaso vs none, HS pts) by decile."""
    n_bins = len(sev_edges) - 1
    cols = [f"D{i + 1}" for i in range(n_bins)]
    fire = {"clinician": 100 * clin_ref["vaso_fire_rate_by_severity"]}
    for c in sorted(results):
        fire[c] = 100 * np.nanmean([r["vaso_fire_rate_by_severity"] for r in results[c]], axis=0)
    fire_df = pd.DataFrame(fire, index=cols).T

    def by_bin(dq, sev):
        return np.array([dq[sev == b].mean() if (sev == b).sum() >= 5 else np.nan for b in range(n_bins)])

    pen = {c: np.nanmean([by_bin(r["dq_vaso_states"], r["dq_vaso_severity"]) for r in results[c]], axis=0)
           for c in sorted(results)}
    if sarsa_ref and sarsa_sev is not None:
        pen["SARSA_mean"] = np.nanmean([by_bin(s["dq_vaso_states"], sarsa_sev) for s in sarsa_ref], axis=0)
    pen_df = pd.DataFrame(pen, index=cols).T
    return fire_df, pen_df