"""
Functions for compare_floor_bound_five_seeds.ipynb.

Compares two 5-seed CQL configurations trained on the AVERAGED Dynamic-DeepHit
reward (identical for every seed) with SEED-SPECIFIC Dynamic-DeepHit states:
  - "no_floor_bound": fluid/vaso mean regularisers 0.05, no vaso floor, no gate bound
  - "floor_bound":    same, plus vaso magnitude floor and vaso gate-logit bound

Every model is evaluated on the processor built for ITS OWN seed (its own state
embedding). Because the reward is shared, Q-values are in the same units for all
seeds and all critics can be compared per patient-hour.

Tiers (agreed before looking at results):
  Tier 1  validity, pass/fail per seed: actor not stuck, actor follows its own critic,
          critic sane (tracks same-seed SARSA, expected sign vs health-score level)
  Tier 2  reproducibility across seeds: critic agreement and policy agreement
  Tier 3  description and independent checks: fluid/vaso behaviour, SARSA advantage
          decomposition, occupancy support (context / tie-break only)
Decision rule: prefer the configuration that passes Tier 1 in all seeds with higher
Tier 2 agreement; if equivalent, prefer the simpler one (no floor/bound). Fluid primary.

Split: validation for everything here; test is held out for final reporting.
"""

import copy
import gc
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import evaluate_vaso_reward_sweep as vrs
from . import diagnose_vaso_saturation as dvs
from .evaluate_cql_alpha_sweep import get_model_dirs, get_critic, compute_occupancy_support_of_actions
from src.rl.evaluate_terminal_reward_models import load_actor_and_critic, load_sarsa_critic, load_data


THR = 1e-4                    # nonzero-dose threshold used throughout the project
SATURATED_RAW = -8.0          # sigmoid'(-8) ~ 3e-4: magnitude head effectively has no gradient
VASO_GRID = np.round(np.arange(0.0, 0.605, 0.01), 3)
FLUID_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 3)
CONFIG_COLORS = {"no_floor_bound": "#2471a3", "floor_bound": "#c0392b"}
CLIN_COLOR = "#7f8c8d"

# Proposed Tier 1 thresholds. Fix these in the notebook BEFORE looking at results.
DEFAULT_THRESHOLDS = {
    "max_vaso_saturated_pct": 5.0,          # % of states with vaso raw mean < -8
    "min_gate_vs_own_critic_rho": 0.3,      # Spearman(vaso gate logit, own-critic vaso gain)
    "max_fired_against_critic_locked_pct": 25.0,   # % of fires with own gain < 0 AND gate p > 0.99
    "min_fluid_actor_vs_critic_rho": 0.3,   # Spearman(actor fluid dose, critic-best fluid dose)
    "min_qfixed_vs_sarsa_rho": 0.5,         # Spearman(state value, same-seed SARSA state value)
    "require_negative_qfixed_vs_health_score": True,   # change-only reward: sicker states have more to gain
}


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return stats.spearmanr(a, b)[0]


# =============================================================================
# 1. Loading and setup checks
# =============================================================================

def _run_seed(model_dir, hp=None):
    """Seed from the run's own checkpoint, then its hyperparameters, then results.json."""
    ck = dvs._checkpoint_hyperparams(model_dir) if os.path.isfile(os.path.join(model_dir, "final_model.pt")) else None
    if ck is not None and ck.get("seed") is not None:
        return int(ck["seed"])
    if hp is not None and hp.get("seed") is not None:
        return int(hp["seed"])
    rj = os.path.join(model_dir, "results.json")
    if os.path.isfile(rj):
        with open(rj) as f:
            s = json.load(f).get("hyperparameters", {}).get("seed")
        if s is not None:
            return int(s)
    raise ValueError(f"No seed found for {model_dir}")


def load_cql_seeds(model_dir, state_dim):
    """{seed: (actor, critic, config_manager, hyperparams)}; seed read from each run's own checkpoint."""
    out = {}
    for d in get_model_dirs(model_dir):
        entry = load_actor_and_critic(model_dir=d, model_type="cql", state_dim=state_dim)
        s = _run_seed(d, entry[3])
        if s in out:
            raise ValueError(f"seed {s} found twice under {model_dir}")
        out[s] = entry
        print(f"  CQL seed {s}: {d}")
    return dict(sorted(out.items()))


def load_sarsa_seeds(model_dir, state_dim):
    """{seed: (critic, config_manager, hyperparams)}."""
    out = {}
    for d in get_model_dirs(model_dir):
        critic, cm, hp = load_sarsa_critic(reward_type="health_score", model_dir=d, state_dim=state_dim)
        s = _run_seed(d, hp)
        if s in out:
            raise ValueError(f"SARSA seed {s} found twice under {model_dir}")
        out[s] = (critic, cm, hp)
        print(f"  SARSA seed {s}: {d}")
    return dict(sorted(out.items()))


KEY_HPARAMS = ["seed", "cql_alpha", "fluid_mean_reg_scale", "vaso_mean_reg_scale",
               "vaso_mean_floor", "vaso_mean_floor_scale", "vaso_gate_logit_bound",
               "vaso_gate_logit_bound_scale", "actor_loss_gate_entropy_scale",
               "add_reward_to_vaso_dosing", "total_health_score_scale", "discount"]


def hyperparameter_table(configs):
    """Key hyperparameters for every (config, seed), read from each run's saved hyperparameters."""
    rows = {}
    for cfg, seeds in configs.items():
        for s, m in seeds.items():
            rows[(cfg, s)] = {k: m[3].get(k) for k in KEY_HPARAMS}
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def hyperparameter_diff(configs, ignore=("seed", "wandb_id", "use_fixed_seed")):
    """All hyperparameters that differ (a) across seeds within a config, (b) between configs (seed 0)."""
    out = {}
    for cfg, seeds in configs.items():
        hps = {s: m[3] for s, m in seeds.items()}
        keys = sorted(set().union(*hps.values()))
        out[f"within {cfg}"] = {k: {s: hps[s].get(k) for s in hps} for k in keys
                                if k not in ignore and len({str(hps[s].get(k)) for s in hps}) > 1}
    cfgs = list(configs)
    if len(cfgs) == 2:
        a, b = (configs[c][min(configs[c])][3] for c in cfgs)
        keys = sorted(set(a) | set(b))
        out["between configs (lowest seed)"] = {k: {cfgs[0]: a.get(k), cfgs[1]: b.get(k)} for k in keys
                                                if k not in ignore and str(a.get(k)) != str(b.get(k))}
    for name, d in out.items():
        print(f"--- {name}: {len(d)} differing keys")
        for k, v in d.items():
            print(f"    {k:40s} {v}")
    return out


def build_seed_processors(base_config_manager, base_hyperparams, seeds, reference_seed=None):
    """
    One validation processor per seed (that seed's Dynamic-DeepHit states, the shared
    averaged reward), plus the TRAIN processor of one reference seed (training actions and
    the averaged health score are identical across seeds, so one train split is enough).
    The base config manager's data paths are used as-is; only training_config['seed'] changes.
    Returns (val_by_seed, train_reference).
    """
    reference_seed = min(seeds) if reference_seed is None else reference_seed
    hp = dict(base_hyperparams)
    hp["add_reward_to_vaso_dosing"] = False
    val_by_seed, train_ref = {}, None
    for s in seeds:
        cm = copy.deepcopy(base_config_manager)
        cm.training_config["seed"] = s
        print(f"  building processors for seed {s} ...", flush=True)
        tr, va, te = load_data(cm, hp)
        val_by_seed[s] = va
        if s == reference_seed:
            train_ref = tr
        else:
            del tr
        del te
        gc.collect()
    print("  data paths used:", base_config_manager.data_config.get("state_path"), "|",
          base_config_manager.data_config.get("reward_path"))
    return val_by_seed, train_ref


def check_seed_processors(val_by_seed):
    """
    Rows must line up across seeds; the reward and health score must be IDENTICAL across
    seeds (averaged reward); the states must DIFFER (seed-specific embeddings).
    """
    seeds = list(val_by_seed)
    ref = val_by_seed[seeds[0]]
    ref_ptr = np.asarray(ref.dynamic_deephit_predictions["index_ptr"])
    ref_act = _np(ref.rl_dataset["actions"])
    ref_hs = np.asarray(ref.dynamic_deephit_predictions["health_score"])
    ref_rew = _np(ref.rl_dataset["rewards"]).ravel() if "rewards" in ref.rl_dataset else None
    ref_st = _np(ref.rl_dataset["states"])
    rows = {}
    for s in seeds:
        p = val_by_seed[s]
        st = _np(p.rl_dataset["states"])
        rew = _np(p.rl_dataset["rewards"]).ravel() if "rewards" in p.rl_dataset else None
        rows[s] = {
            "n_states": len(st),
            "same_index_ptr": np.array_equal(np.asarray(p.dynamic_deephit_predictions["index_ptr"]), ref_ptr),
            "same_actions": st.shape[0] == ref_act.shape[0] and np.allclose(_np(p.rl_dataset["actions"]), ref_act),
            "same_health_score": np.allclose(np.asarray(p.dynamic_deephit_predictions["health_score"]), ref_hs),
            "same_rewards": (rew is not None and ref_rew is not None and np.allclose(rew, ref_rew)),
            "states_identical_to_first": st.shape == ref_st.shape and np.allclose(st, ref_st),
        }
    df = pd.DataFrame(rows).T
    df.index.name = "seed"
    return df


def identify_state_seed(models_by_seed, val_by_seed, gamma=None):
    """
    Bellman-consistency check: median |Q(s,a) - (r + gamma Q(s',a'))| / sd(Q) for each model's
    critic on each seed's processor (clinician next actions). The diagonal (own states)
    should be the smallest in each row; anything else means a model/state mismatch.
    """
    rows = {}
    for ms, m in models_by_seed.items():
        g = gamma if gamma is not None else m[3].get("discount", 0.975)
        r = {}
        for ps, proc in val_by_seed.items():
            br = dvs.bellman_residual(m[1], proc, g)
            r[f"states seed {ps}"] = br["median_abs_residual"] / max(br["Q_sd"], 1e-8)
        r["best_match_seed"] = min(val_by_seed, key=lambda ps: r[f"states seed {ps}"])
        r["matches_own_seed"] = r["best_match_seed"] == ms
        rows[f"model seed {ms}"] = r
    return pd.DataFrame(rows).T


# =============================================================================
# Shared computations (done once per model and cached)
# =============================================================================

def actor_outputs(model_entry, processor):
    """Raw actor parameters and deterministic actions on the model's own states."""
    p = dvs.actor_raw_params(model_entry[0], processor.rl_dataset["states"])
    f_dose = (p["f_p"] > 0.5) * dvs._sigmoid(p["f_raw_mean"])
    v_dose = (p["v_p"] > 0.5) * dvs._sigmoid(p["v_raw_mean"])
    p["actions"] = np.stack([f_dose, v_dose], axis=1).astype(np.float32)
    return p


def compute_all_actor_outputs(configs, val_by_seed):
    """{config: {seed: actor_outputs}}"""
    return {cfg: {s: actor_outputs(m, val_by_seed[s]) for s, m in seeds.items()} for cfg, seeds in configs.items()}


def _q(critic, states, actions):
    return vrs.critic_q(critic, states, np.asarray(actions, dtype=np.float32))


def q_curve(critic, scale, states, base_actions, drug_idx, grid):
    """Q(s, base action with one drug set to each grid dose), HS points. Shape (n, len(grid))."""
    base = np.asarray(base_actions, dtype=np.float32)
    out = np.empty((len(states), len(grid)), dtype=np.float64)
    for j, d in enumerate(grid):
        a = base.copy()
        a[:, drug_idx] = d
        out[:, j] = _q(critic, states, a) / scale
    return out


def critic_quantities(critic, scale, processor, fixed_action=(0.5, 0.0), delta=0.1, fluid_abs_dose=None):
    """
    Per-state critic quantities on the model's own states, all in HS points:
      q_fixed      Q(s, fixed action)                      -- state-value proxy
      q_clin       Q(s, a_clin)
      vaso_gain    Q(s, [f_clin, 0.1]) - Q(s, [f_clin, 0])
      fluid_local  Q(s, f_clin + delta) - Q(s, f_clin - delta), vaso at clinician's
      fluid_abs    Q(s, [d, v_clin]) - Q(s, [0, v_clin]), d = median clinician nonzero fluid
    """
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"]).astype(np.float32)
    if fluid_abs_dose is None:
        fluid_abs_dose = float(np.median(clin[clin[:, 0] > THR, 0]))
    fixed = np.tile(np.asarray(fixed_action, dtype=np.float32), (len(clin), 1))

    def with_dose(j, d):
        a = clin.copy()
        a[:, j] = d
        return a

    q = {"q_fixed": _q(critic, states, fixed) / scale,
         "q_clin": _q(critic, states, clin) / scale}
    q["vaso_gain"] = (_q(critic, states, with_dose(1, 0.1)) - _q(critic, states, with_dose(1, 0.0))) / scale
    hi, lo = clin.copy(), clin.copy()
    hi[:, 0] = np.clip(clin[:, 0] + delta, 0, 1)
    lo[:, 0] = np.clip(clin[:, 0] - delta, 0, 1)
    q["fluid_local"] = (_q(critic, states, hi) - _q(critic, states, lo)) / scale
    q["fluid_abs"] = (_q(critic, states, with_dose(0, fluid_abs_dose)) - _q(critic, states, with_dose(0, 0.0))) / scale
    return q


def compute_all_critic_quantities(configs, sarsa_by_seed, val_by_seed):
    """{set_name: {seed: critic_quantities}} for each CQL config and for SARSA."""
    out = {}
    for cfg, seeds in configs.items():
        out[cfg] = {}
        for s, m in seeds.items():
            print(f"  critic quantities: {cfg} seed {s}", flush=True)
            out[cfg][s] = critic_quantities(m[1], vrs._reward_scale(m[3], cfg), val_by_seed[s])
    if sarsa_by_seed:
        out["SARSA"] = {}
        for s, (c, cm, hp) in sarsa_by_seed.items():
            print(f"  critic quantities: SARSA seed {s}", flush=True)
            out["SARSA"][s] = critic_quantities(get_critic((c, cm, hp)), vrs._reward_scale(hp, "SARSA"), val_by_seed[s])
    return out


# =============================================================================
# 2. Tier 1a: actor health
# =============================================================================

def actor_health_table(actor_out, floor=None):
    """Per (config, seed): is either magnitude head or gate stuck?"""
    rows = {}
    smin, smax = np.exp(-2.0), np.exp(1.0)
    for cfg, seeds in actor_out.items():
        for s, p in seeds.items():
            vr, vp, fr, fp = p["v_raw_mean"], p["v_p"], p["f_raw_mean"], p["f_p"]
            fired = p["actions"][:, 1] > THR
            r = {
                "vaso_raw_median": np.median(vr),
                "vaso_raw_p10": np.percentile(vr, 10),
                "vaso_raw_p90": np.percentile(vr, 90),
                "vaso_saturated_%": 100 * (vr < SATURATED_RAW).mean(),
                "vaso_fire_%": 100 * fired.mean(),
                "vaso_gate_p_sd": vp.std(),
                "vaso_gate_p<0.01_%": 100 * (vp < 0.01).mean(),
                "vaso_fired_with_p>0.99_%": 100 * (vp[fired] > 0.99).mean() if fired.any() else np.nan,
                "vaso_std_at_min_%": 100 * np.isclose(p["v_std"], smin, atol=1e-3).mean(),
                "vaso_std_at_max_%": 100 * np.isclose(p["v_std"], smax, atol=1e-3).mean(),
                "fluid_raw_median": np.median(fr),
                "fluid_saturated_%": 100 * (np.abs(fr) > -SATURATED_RAW).mean(),
                "fluid_fire_%": 100 * (p["actions"][:, 0] > THR).mean(),
                "fluid_gate_p>0.99_%": 100 * (fp > 0.99).mean(),
                "fluid_std_at_min_%": 100 * np.isclose(p["f_std"], smin, atol=1e-3).mean(),
                "fluid_std_at_max_%": 100 * np.isclose(p["f_std"], smax, atol=1e-3).mean(),
            }
            if floor is not None:
                r["vaso_below_floor_%"] = 100 * (vr < floor).mean()
            rows[(cfg, s)] = r
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def plot_vaso_raw_means(actor_out, floor=None, save_path=None):
    """Distribution of the vaso magnitude mean (raw logit) per seed, one panel per config."""
    cfgs = list(actor_out)
    fig, axes = plt.subplots(1, len(cfgs), figsize=(6.5 * len(cfgs), 4), squeeze=False)
    allv = np.concatenate([p["v_raw_mean"] for seeds in actor_out.values() for p in seeds.values()])
    bins = np.linspace(np.percentile(allv, 0.5) - 0.5, np.percentile(allv, 99.5) + 0.5, 80)
    cmap = plt.get_cmap("viridis")
    for ax, cfg in zip(axes[0], cfgs):
        seeds = sorted(actor_out[cfg])
        for i, s in enumerate(seeds):
            ax.hist(actor_out[cfg][s]["v_raw_mean"], bins=bins, histtype="step", lw=1.4,
                    color=cmap(i / max(1, len(seeds) - 1)), density=True, label=f"seed {s}")
        ax.axvline(SATURATED_RAW, color="k", ls=":", lw=1, label="saturated (< -8)")
        if floor is not None:
            ax.axvline(floor, color="#c0392b", ls="--", lw=1, label=f"floor {floor:g}")
        ax.set_title(cfg, fontsize=10)
        ax.set_xlabel("vaso magnitude mean (raw logit)")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# 3. Tier 1b/c: actor follows its own critic; critic sanity
# =============================================================================

def actor_follows_critic(model_entry, out, processor, cfg_name=""):
    """
    On the model's own states, with the actor's OWN fluid/vaso as context:
      vaso gate vs own-critic gain (dose 0.1 vs 0), vaso dose vs critic-best dose at fired states,
      fluid dose vs critic-best fluid dose at all states.
    """
    critic, hp = model_entry[1], model_entry[3]
    scale = vrs._reward_scale(hp, cfg_name)
    states = _np(processor.rl_dataset["states"])
    pol = out["actions"]
    vp = out["v_p"]
    gate_logit = dvs._logit(vp, eps=1e-6)
    fired = pol[:, 1] > THR

    base = pol.copy()
    base[:, 1] = 0.1
    q01 = _q(critic, states, base)
    base[:, 1] = 0.0
    q00 = _q(critic, states, base)
    gain = (q01 - q00) / scale
    pos = gain > 0
    against = fired & ~pos

    r = {
        "own_vaso_gain>0_%": 100 * pos.mean(),
        "gate_vs_own_critic_rho": _spearman(gate_logit, gain),
        "gate_auroc_vs_gain>0": roc_auc_score(pos, vp) if 0 < pos.mean() < 1 else np.nan,
        "fired_against_critic_%": 100 * against.sum() / max(fired.sum(), 1) if fired.any() else np.nan,
        "fired_against_critic_locked_%": (100 * (against & (vp > 0.99)).sum() / max(fired.sum(), 1)
                                          if fired.any() else np.nan),
        "not_fired_where_gain>0_%": 100 * (~fired & pos).sum() / max(pos.sum(), 1),
    }

    if fired.sum() >= 10:
        idx = np.where(fired)[0]
        curves = q_curve(critic, scale, states[idx], pol[idx], 1, VASO_GRID)
        curves -= curves[:, [0]]
        best = VASO_GRID[curves.argmax(1)]
        q_actor, _ = dvs._interp_rows(pol[idx, 1].astype(np.float64), VASO_GRID.astype(np.float64), curves)
        r.update({
            "vaso_dose_vs_best_rho": _spearman(pol[idx, 1], best),
            "vaso_dose_median": np.median(pol[idx, 1]),
            "vaso_best_dose_median": np.median(best),
            "vaso_regret_median_HS": np.median(curves.max(1) - q_actor),
        })
    else:
        r.update({k: np.nan for k in ["vaso_dose_vs_best_rho", "vaso_dose_median",
                                      "vaso_best_dose_median", "vaso_regret_median_HS"]})

    fcurves = q_curve(critic, scale, states, pol, 0, FLUID_GRID)
    fbest = FLUID_GRID[fcurves.argmax(1)]
    fq_actor, _ = dvs._interp_rows(pol[:, 0].astype(np.float64), FLUID_GRID.astype(np.float64), fcurves)
    r.update({
        "fluid_dose_vs_best_rho": _spearman(pol[:, 0], fbest),
        "fluid_dose_median": np.median(pol[:, 0]),
        "fluid_best_dose_median": np.median(fbest),
        "fluid_best_is_0_%": 100 * (fbest == 0).mean(),
        "fluid_regret_median_HS": np.median(fcurves.max(1) - fq_actor),
    })
    return r


def actor_follows_critic_table(configs, actor_out, val_by_seed):
    rows = {}
    for cfg, seeds in configs.items():
        for s, m in seeds.items():
            print(f"  actor-vs-critic: {cfg} seed {s}", flush=True)
            rows[(cfg, s)] = actor_follows_critic(m, actor_out[cfg][s], val_by_seed[s], cfg)
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def critic_sanity_table(critic_q, health_score):
    """
    Per (set, seed): Spearman(state value, health score) -- expected NEGATIVE for a
    change-only reward; Spearman(state value, same-seed SARSA) and vs the SARSA ensemble
    (each SARSA critic scored on its own states); share of Q variance due to the action.
    """
    hs = np.asarray(health_score)
    sarsa = critic_q.get("SARSA", {})
    sarsa_mean = np.mean([q["q_fixed"] for q in sarsa.values()], axis=0) if sarsa else None
    rows = {}
    for name, seeds in critic_q.items():
        for s, q in seeds.items():
            qf, qc = q["q_fixed"], q["q_clin"]
            rows[(name, s)] = {
                "qfixed_vs_health_score_rho": _spearman(qf, hs),
                "qfixed_vs_same_seed_sarsa_rho": (_spearman(qf, sarsa[s]["q_fixed"])
                                                  if s in sarsa and name != "SARSA" else np.nan),
                "qfixed_vs_sarsa_ensemble_rho": _spearman(qf, sarsa_mean) if sarsa_mean is not None else np.nan,
                "action_share_of_Q_var": (qc - qf).var() / qc.var(),
                "q_clin_mean_HS": qc.mean(),
                "q_clin_IQR_HS": np.subtract(*np.percentile(qc, [75, 25])),
            }
    df = pd.DataFrame(rows).T
    df.index.names = ["set", "seed"]
    return df


# =============================================================================
# 4. Tier 2: agreement across seeds
# =============================================================================

def _pairwise(arrs, mask=None):
    keys = list(arrs)
    vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = arrs[keys[i]], arrs[keys[j]]
            if mask is not None:
                a, b = a[mask], b[mask]
            vals.append(_spearman(a, b))
    return np.array(vals)


def critic_agreement_table(critic_q, clinician_actions):
    """
    Across seeds, per critic set: mean/min/max pairwise Spearman of each per-state quantity
    (vaso gain also on clinician-vaso states); sign agreement vs what independent critics
    with the same base rates would give; share of states where the ensemble mean, all
    critics, and the ensemble lower 95% bound show a positive gain.
    """
    clin = _np(clinician_actions)
    clin_v = clin[:, 1] > THR
    rows = {}
    for name, seeds in critic_q.items():
        r = {}
        for qname in ["q_fixed", "q_clin", "vaso_gain", "fluid_local", "fluid_abs"]:
            arrs = {s: q[qname] for s, q in seeds.items()}
            pw = _pairwise(arrs)
            r[f"{qname}: rho mean"] = np.nanmean(pw)
            r[f"{qname}: rho min"] = np.nanmin(pw)
            if qname == "vaso_gain":
                r["vaso_gain (clin-vaso states): rho mean"] = np.nanmean(_pairwise(arrs, clin_v))
            if qname in ("vaso_gain", "fluid_local", "fluid_abs"):
                G = np.stack(list(arrs.values()))
                pos = G > 0
                p_pos = pos.mean(1)
                lcb = G.mean(0) - 1.96 * G.std(0, ddof=1) / np.sqrt(len(G))
                r[f"{qname}: sign agree obs %"] = 100 * (pos.all(0) | (~pos).all(0)).mean()
                r[f"{qname}: sign agree if indep %"] = 100 * (np.prod(p_pos) + np.prod(1 - p_pos))
                r[f"{qname}: ensemble mean>0 %"] = 100 * (G.mean(0) > 0).mean()
                r[f"{qname}: all>0 %"] = 100 * pos.all(0).mean()
                r[f"{qname}: ensemble LCB>0 %"] = 100 * (lcb > 0).mean()
        rows[name] = r
    return pd.DataFrame(rows).T


def policy_agreement_table(actor_out, clinician_actions, train_actions):
    """
    Across seeds, per config: fluid dose agreement, fluid fire agreement, vaso firing overlap
    (Jaccard) at any dose and at a clinical dose (clinician p5), and how many states get vaso
    from k seeds.
    """
    tv = _np(train_actions)[:, 1]
    p5 = float(np.percentile(tv[tv > THR], 5))
    rows, counts = {}, {}
    for cfg, seeds in actor_out.items():
        acts = {s: p["actions"] for s, p in seeds.items()}
        f = {s: a[:, 0] for s, a in acts.items()}
        F = np.stack(list(f.values()))
        r = {"fluid_dose: rho mean": np.nanmean(_pairwise(f)),
             "fluid_dose: rho min": np.nanmin(_pairwise(f)),
             "fluid_dose: per-state SD across seeds (median)": np.median(F.std(0)),
             "fluid_fire: all seeds agree %": 100 * ((F > THR).all(0) | (F <= THR).all(0)).mean()}
        for label, thr in [("any", THR), ("clin p5", p5)]:
            V = np.stack([a[:, 1] > thr for a in acts.values()])
            jac = []
            for i in range(len(V)):
                for j in range(i + 1, len(V)):
                    u = (V[i] | V[j]).sum()
                    jac.append((V[i] & V[j]).sum() / u if u else np.nan)
            r[f"vaso fire ({label}): Jaccard mean"] = np.nanmean(jac) if jac else np.nan
            r[f"vaso fire ({label}): fired by >=1 seed %"] = 100 * V.any(0).mean()
            r[f"vaso fire ({label}): fired by all seeds %"] = 100 * V.all(0).mean()
            counts[(cfg, label)] = pd.Series(V.sum(0)).value_counts().sort_index()
        rows[cfg] = r
    count_df = pd.DataFrame(counts).fillna(0).astype(int)
    count_df.index.name = "n_seeds_firing"
    return pd.DataFrame(rows).T, count_df


# =============================================================================
# 5. Tier 3: policy description
# =============================================================================

def policy_description_table(actor_out, clinician_actions, train_actions):
    """Per (config, seed) fluid and vaso behaviour, plus a clinician row."""
    clin = _np(clinician_actions)
    clin_v, clin_f = clin[:, 1] > THR, clin[:, 0] > THR
    tv = _np(train_actions)[:, 1]
    p5 = float(np.percentile(tv[tv > THR], 5))
    rows = {("clinician", "-"): {
        "fluid_fire_%": 100 * clin_f.mean(), "fluid_dose_mean": clin[:, 0].mean(),
        "fluid_dose_when_fired": clin[clin_f, 0].mean(),
        "vaso_fire_%": 100 * clin_v.mean(), "vaso_fire_clin_p5_%": 100 * (clin[:, 1] > p5).mean(),
        "vaso_dose_when_fired": clin[clin_v, 1].mean()}}
    for cfg, seeds in actor_out.items():
        for s, p in seeds.items():
            a = p["actions"]
            f_f, v_f = a[:, 0] > THR, a[:, 1] > THR
            rows[(cfg, s)] = {
                "fluid_fire_%": 100 * f_f.mean(),
                "fluid_dose_mean": a[:, 0].mean(),
                "fluid_dose_when_fired": a[f_f, 0].mean() if f_f.any() else np.nan,
                "fluid_vs_clinician_rho": _spearman(a[:, 0], clin[:, 0]),
                "fluid_minus_clinician_mean": (a[:, 0] - clin[:, 0]).mean(),
                "vaso_fire_%": 100 * v_f.mean(),
                "vaso_fire_clin_p5_%": 100 * (a[:, 1] > p5).mean(),
                "vaso_fire_at_clin_vaso_%": 100 * v_f[clin_v].mean(),
                "vaso_fire_at_clin_zero_%": 100 * v_f[~clin_v].mean(),
                "vaso_PPV_%": 100 * (v_f & clin_v).sum() / max(v_f.sum(), 1) if v_f.any() else np.nan,
                "vaso_gate_auroc_vs_clin": roc_auc_score(clin_v, p["v_p"]) if 0 < clin_v.mean() < 1 else np.nan,
                "vaso_dose_when_fired": a[v_f, 1].mean() if v_f.any() else np.nan,
            }
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def severity_profiles(actor_out, clinician_actions, health_score, sev_edges):
    """Mean fluid dose and vaso fire rate (%) by health-score decile (D1 = sickest), per (config, seed) + clinician."""
    clin = _np(clinician_actions)
    sev = vrs.severity_bin(np.asarray(health_score), sev_edges)
    nb = len(sev_edges) - 1
    cols = [f"D{i + 1}" for i in range(nb)]

    def prof(a):
        f = [a[sev == b, 0].mean() if (sev == b).any() else np.nan for b in range(nb)]
        v = [100 * (a[sev == b, 1] > THR).mean() if (sev == b).any() else np.nan for b in range(nb)]
        return f, v

    fl, va = {}, {}
    fl[("clinician", "-")], va[("clinician", "-")] = prof(clin)
    for cfg, seeds in actor_out.items():
        for s, p in seeds.items():
            fl[(cfg, s)], va[(cfg, s)] = prof(p["actions"])
    fdf = pd.DataFrame(fl, index=cols).T
    vdf = pd.DataFrame(va, index=cols).T
    for d in (fdf, vdf):
        d.index.names = ["config", "seed"]
    return fdf, vdf


def plot_severity_profiles(fluid_df, vaso_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(1, fluid_df.shape[1] + 1)
    for ax, df, ylab in [(axes[0], fluid_df, "mean fluid dose"), (axes[1], vaso_df, "vaso fire rate (%)")]:
        for (cfg, s), row in df.iterrows():
            if cfg == "clinician":
                ax.plot(x, row.values, color=CLIN_COLOR, lw=2.5, label="clinician", zorder=5)
            else:
                ax.plot(x, row.values, color=CONFIG_COLORS.get(cfg, "k"), lw=1, alpha=0.6,
                        label=cfg if s == min(df.loc[cfg].index) else None)
        ax.set_xlabel("health-score decile (1 = sickest)")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Fluid by severity (thin lines = seeds)", fontsize=10)
    axes[1].set_title("Vaso by severity (thin lines = seeds)", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# 6. Tier 3: independent checks
# =============================================================================

def sarsa_advantage_table(actor_out, sarsa_by_seed, val_by_seed):
    """
    Each CQL seed's deterministic actions judged by EVERY SARSA critic, each SARSA critic on
    its own states (rows align, actions are physical doses). Split into a fluid part
    (policy with vaso removed vs clinician) and a vaso part (policy vs policy without vaso),
    normalised by each SARSA critic's Q IQR at clinician actions; mean over SARSA seeds.
    SARSA critics carry the same confounding: context, not proof.
    """
    sar = {}
    for j, (c, cm, hp) in sarsa_by_seed.items():
        proc = val_by_seed[j]
        st = _np(proc.rl_dataset["states"])
        clin = _np(proc.rl_dataset["actions"]).astype(np.float32)
        crit = get_critic((c, cm, hp))
        q_clin = _q(crit, st, clin)
        sar[j] = (crit, st, q_clin, np.subtract(*np.percentile(q_clin, [75, 25])))
    rows = {}
    for cfg, seeds in actor_out.items():
        for s, p in seeds.items():
            pol = p["actions"]
            nov = pol.copy()
            nov[:, 1] = 0.0
            tot, flu, vas = [], [], []
            for j, (crit, st, q_clin, iqr) in sar.items():
                q_pol, q_nov = _q(crit, st, pol), _q(crit, st, nov)
                tot.append(((q_pol - q_clin) / iqr).mean())
                flu.append(((q_nov - q_clin) / iqr).mean())
                vas.append(((q_pol - q_nov) / iqr).mean())
            rows[(cfg, s)] = {"sarsa_adv_total": np.mean(tot), "sarsa_adv_fluid_part": np.mean(flu),
                              "sarsa_adv_vaso_part": np.mean(vas), "sarsa_adv_total_sd_over_judges": np.std(tot)}
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def occupancy_table(actor_out, health_score, train_actions, train_health_score, n_severity_bins=200, min_count=10):
    """Fraction of each policy's actions that are well supported by training data (same bins as before)."""
    rows = {}
    for cfg, seeds in actor_out.items():
        for s, p in seeds.items():
            frac, _ = compute_occupancy_support_of_actions(
                p["actions"], np.asarray(health_score), _np(train_actions), np.asarray(train_health_score),
                n_severity_bins=n_severity_bins, min_count=min_count)
            rows[(cfg, s)] = {"occupancy_support_%": 100 * frac}
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


# =============================================================================
# 7. Summary and decision
# =============================================================================

def seed_spread(per_seed_df):
    """Mean / SD / min / max across seeds for every numeric column, per config (or set)."""
    num = per_seed_df.apply(pd.to_numeric, errors="coerce")
    return num.groupby(level=0, sort=False).agg(["mean", "std", "min", "max"])


def tier1_checks(health_df, follows_df, sanity_df, thresholds=DEFAULT_THRESHOLDS):
    """Pass/fail per (config, seed) for each Tier 1 criterion, using thresholds fixed in advance."""
    t = thresholds
    rows = {}
    for key in health_df.index:
        cfg, s = key
        h, f = health_df.loc[key], follows_df.loc[key]
        san = sanity_df.loc[(cfg, s)] if (cfg, s) in sanity_df.index else None
        locked = f["fired_against_critic_locked_%"]
        r = {
            "vaso_not_saturated": h["vaso_saturated_%"] < t["max_vaso_saturated_pct"],
            "gate_follows_own_critic": (f["gate_vs_own_critic_rho"] > t["min_gate_vs_own_critic_rho"])
                                       if not np.isnan(f["gate_vs_own_critic_rho"]) else False,
            "gate_not_locked_against_critic": True if np.isnan(locked) else locked < t["max_fired_against_critic_locked_pct"],
            "fluid_follows_own_critic": f["fluid_dose_vs_best_rho"] > t["min_fluid_actor_vs_critic_rho"],
        }
        if san is not None:
            r["critic_tracks_same_seed_sarsa"] = san["qfixed_vs_same_seed_sarsa_rho"] > t["min_qfixed_vs_sarsa_rho"]
            if t.get("require_negative_qfixed_vs_health_score", True):
                r["critic_expected_sign_vs_health_score"] = san["qfixed_vs_health_score_rho"] < 0
        r["ALL_PASS"] = all(r.values())
        rows[key] = r
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def decision_summary(tier1_df, critic_agree_df, policy_agree_df, description_df, sarsa_df=None, occupancy_df=None):
    """One table: Tier 1 pass counts, Tier 2 agreement, Tier 3 means across seeds, per config."""
    cfgs = [c for c in tier1_df.index.get_level_values(0).unique()]
    rows = {}
    for cfg in cfgs:
        t1 = tier1_df.loc[cfg]
        r = {f"T1 {c}: seeds passing": f"{int(t1[c].sum())}/{len(t1)}" for c in t1.columns}
        ca = critic_agree_df.loc[cfg]
        for k in ["q_fixed: rho mean", "fluid_local: rho mean", "fluid_abs: rho mean",
                  "vaso_gain: rho mean", "vaso_gain (clin-vaso states): rho mean"]:
            r[f"T2 critic {k}"] = ca[k]
        pa = policy_agree_df.loc[cfg]
        for k in ["fluid_dose: rho mean", "fluid_fire: all seeds agree %",
                  "vaso fire (any): Jaccard mean", "vaso fire (clin p5): Jaccard mean"]:
            r[f"T2 policy {k}"] = pa[k]
        d = description_df.loc[cfg].apply(pd.to_numeric, errors="coerce")
        for k in ["fluid_fire_%", "fluid_dose_mean", "vaso_fire_%", "vaso_fire_clin_p5_%", "vaso_PPV_%"]:
            r[f"T3 {k} (mean±sd)"] = f"{d[k].mean():.3g} ± {d[k].std():.2g}"
        if sarsa_df is not None:
            sd = sarsa_df.loc[cfg]
            for k in ["sarsa_adv_fluid_part", "sarsa_adv_vaso_part"]:
                r[f"T3 {k} (mean±sd)"] = f"{sd[k].mean():.3g} ± {sd[k].std():.2g}"
        if occupancy_df is not None:
            od = occupancy_df.loc[cfg]["occupancy_support_%"]
            r["T3 occupancy_support_% (mean±sd)"] = f"{od.mean():.3g} ± {od.std():.2g}"
        rows[cfg] = r
    return pd.DataFrame(rows)


# =============================================================================
# Dose levels: policy vs clinician, both drugs
# =============================================================================

def dose_level_table(actor_out, clinician_actions):
    """
    Per (config, seed) and clinician, for each drug: fire rate, mean dose over all states,
    and the distribution of NONZERO doses (mean, p10, median, p90). Also the mean dose at
    the states where the clinician gave that drug. Doses are on the normalised
    (log1p, min-max) scale used for training.
    """
    clin = _np(clinician_actions)

    def stats_for(a, clin_given):
        out = {}
        for j, drug in [(0, "fluid"), (1, "vaso")]:
            d = a[:, j]
            nz = d[d > THR]
            out[f"{drug}_fire_%"] = 100 * (d > THR).mean()
            out[f"{drug}_mean_all"] = d.mean()
            out[f"{drug}_nonzero_mean"] = nz.mean() if len(nz) else np.nan
            out[f"{drug}_nonzero_p10"] = np.percentile(nz, 10) if len(nz) else np.nan
            out[f"{drug}_nonzero_median"] = np.median(nz) if len(nz) else np.nan
            out[f"{drug}_nonzero_p90"] = np.percentile(nz, 90) if len(nz) else np.nan
            out[f"{drug}_mean_at_clin_given"] = d[clin_given[j]].mean()
        return out

    clin_given = [clin[:, 0] > THR, clin[:, 1] > THR]
    rows = {("clinician", "-"): stats_for(clin, clin_given)}
    for cfg, seeds in actor_out.items():
        for s, p in seeds.items():
            rows[(cfg, s)] = stats_for(p["actions"], clin_given)
    df = pd.DataFrame(rows).T
    df.index.names = ["config", "seed"]
    return df


def plot_dose_distributions(actor_out, clinician_actions, save_path=None):
    """Nonzero-dose distributions (density): clinician vs each config (thin line per seed), per drug."""
    clin = _np(clinician_actions)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (j, drug) in zip(axes, [(0, "fluid"), (1, "vaso")]):
        bins = np.linspace(0, 1, 51)
        c_nz = clin[clin[:, j] > THR, j]
        ax.hist(c_nz, bins=bins, density=True, color=CLIN_COLOR, alpha=0.45,
                label=f"clinician (fires {100 * (clin[:, j] > THR).mean():.1f}%)")
        for cfg, seeds in actor_out.items():
            for k, (s, p) in enumerate(sorted(seeds.items())):
                d = p["actions"][:, j]
                nz = d[d > THR]
                if len(nz) < 10:
                    continue
                ax.hist(nz, bins=bins, density=True, histtype="step", lw=1.2, alpha=0.8,
                        color=CONFIG_COLORS.get(cfg, "k"), label=cfg if k == 0 else None)
        ax.set_xlabel(f"{drug} dose when given (normalised)")
        ax.set_ylabel("density")
        ax.set_title(f"{drug}: nonzero doses (thin lines = seeds)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig