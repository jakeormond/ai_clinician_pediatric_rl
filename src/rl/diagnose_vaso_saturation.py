"""
Diagnosing (and verifying the fix for) vaso magnitude-head saturation in the
health-score CQL actor.

Finding that motivated this (vaso reward sweep, c=0, val):
  - v_raw_mean ~ -12.7 at every clinician-vaso state (sigmoid'(x) ~ 3e-6),
    identical whether the own critic favours vaso (dQ>0, 55% of states) or not
  - vaso gate p ~ 0.017 everywhere -> the actor's vaso output is a
    state-independent constant; it is not reading the critic
Mechanism:
  - the vaso mean-saturation regulariser is gate-weighted
    (0.5 - |p - 0.5|), i.e. ~p ~ 0.017 when the gate is near-closed -> off
  - the magnitude head only receives Q-gradient on FIRED samples
    (action = fire * mag), ~1.7% of the batch, and there sigmoid' ~ 3e-6
  -> once v_raw_mean drifts low, neither head can recover.
Fix (cql_with_logprobs.py): an un-gated one-sided hinge
    vaso_mean_floor_scale * relu(vaso_mean_floor - v_raw_mean)^2
with `vaso_mean_floor` a hyperparameter derived here from clinician data.

Workflow:
  1. recommend_vaso_floor              -> value to put in the training config
                                          (percentile-based; the min-dose version is
                                          outlier-driven and leaves the gate without signal)
  2. saturation_table / plots          -> how widespread is it in existing models?
  3. gate_vs_critic_table              -> does each actor follow its own critic?
  4. (after retraining) repeat 2-3 on floor runs + vrs behaviour tables
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from scipy import stats

from .evaluate_cql_alpha_sweep import get_model_dirs
from src.rl.evaluate_terminal_reward_models import load_actor_and_critic
from . import evaluate_vaso_reward_sweep as vrs


VASO_THRESHOLD = 1e-4
LOG_STD_MIN, LOG_STD_MAX = -2.0, 1.0          # Actor defaults (clamp on log std)
SATURATED_RAW = -8.0                          # sigmoid'(-8) ~ 3.4e-4: effectively no gradient


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# 1. Choosing the floor
# =============================================================================

def recommend_vaso_floor(training_actions, method="percentile", pct=1.0, n_sd=3.0,
                         threshold=VASO_THRESHOLD):
    """
    Floor for the vaso magnitude mean in raw (logit) space.
      method="percentile": pct-th percentile of clinician nonzero doses in raw space (default)
      method="sd":         raw_mean - n_sd * raw_std
      method="min_sd":     logit(min dose) - n_sd * raw_std  (outlier-driven; not recommended --
                           on the CHOC data the min dose is ~6.4 SD below the mean, giving a
                           floor at dose ~0.003)
    The floor must sit at a dose the critic can distinguish from zero: the gate's
    Q-gradient scales with the dose it gates (action = fire * mag), so a floor at a
    near-zero dose releases the magnitude head but leaves the gate without signal.
    """
    v = _np(training_actions)[:, 1]
    nz = v[v > threshold]
    raw = _logit(nz)
    if method == "percentile":
        floor = np.percentile(raw, pct)
    elif method == "sd":
        floor = raw.mean() - n_sd * raw.std()
    elif method == "min_sd":
        floor = _logit(nz.min()) - n_sd * raw.std()
    else:
        raise ValueError(f"unknown method {method}")

    def g(x):
        return _sigmoid(x) * (1 - _sigmoid(x))

    out = {
        "method": f"{method} (pct={pct})" if method == "percentile" else f"{method} (n_sd={n_sd})",
        "n_nonzero_clinician": len(nz),
        "min_nonzero_dose": float(nz.min()),
        "min_dose_z_score": float((_logit(nz.min()) - raw.mean()) / raw.std()),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std()),
        "raw_p0.1 / p1 / p5": " / ".join(f"{np.percentile(raw, q):.3f}" for q in (0.1, 1, 5)),
        "vaso_mean_floor": float(floor),
        "dose_at_floor": float(_sigmoid(floor)),
        "sigmoid_grad_at_floor": float(g(floor)),
        "grad_ratio_floor_vs_clin_mean": float(g(floor) / g(raw.mean())),
        "frac_clinician_doses_below_floor_%": float(100 * (raw < floor).mean()),
    }
    for k, val in out.items():
        print(f"  {k:38s} {val:.4g}" if isinstance(val, float) else f"  {k:38s} {val}")
    print(f"\n  -> set in training config:  'vaso_mean_floor': {floor:.3f},  'vaso_mean_floor_scale': 0.05")
    return out


def plot_clinician_vaso_raw(training_actions, floor, model_raw_means=None, save_path=None):
    """Clinician nonzero vaso doses in raw (logit) space, with the floor and,
    optionally, model v_raw_mean distributions ({label: array}) overlaid."""
    v = _np(training_actions)[:, 1]
    raw = _logit(v[v > VASO_THRESHOLD])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo = min(raw.min(), floor, *(np.min(a) for a in (model_raw_means or {}).values())) - 1
    bins = np.linspace(lo, raw.max() + 1, 80)
    ax.hist(raw, bins=bins, density=True, color="#7f8c8d", alpha=0.6, label="clinician (nonzero)")
    for label, arr in (model_raw_means or {}).items():
        ax.hist(arr, bins=bins, density=True, histtype="step", lw=1.5, label=label)
    ax.axvline(floor, color="#c0392b", ls="--", label=f"floor = {floor:.2f} (dose {_sigmoid(floor):.3f})")
    ax.axvline(SATURATED_RAW, color="k", ls=":", lw=1, label=f"saturated (< {SATURATED_RAW:g})")
    ax.set_xlabel("raw vaso magnitude (logit space)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# Loading helpers
# =============================================================================

def effective_vaso_coef(hp):
    """vaso_reward_coefficient only takes effect when add_reward_to_vaso_dosing is
    True (older runs sometimes set the coefficient with the flag off)."""
    if not hp.get("add_reward_to_vaso_dosing", False):
        return 0.0
    return float(hp.get("vaso_reward_coefficient", 0.0))


def load_models(model_dir, state_dim, label_fmt="seed {seed}"):
    """Loads every run under model_dir -> {label: model_entry}.
    {coef} in label_fmt is the EFFECTIVE vaso reward coefficient (0 unless
    add_reward_to_vaso_dosing is True)."""
    out = {}
    for d in get_model_dirs(model_dir):
        entry = load_actor_and_critic(model_dir=d, model_type="cql", state_dim=state_dim)
        hp = entry[3]
        label = label_fmt.format(seed=hp.get("seed"), alpha=hp.get("cql_alpha"),
                                 coef=f"{effective_vaso_coef(hp):g}")
        if label in out:
            label = f"{label} ({os.path.basename(d)})"
        out[label] = entry
    return out


def load_matching_alpha_run(alpha_sweep_dir, target_alpha, state_dim):
    """The alpha-sweep run with cql_alpha == target_alpha (matched from checkpoint)."""
    matches = []
    for d in get_model_dirs(alpha_sweep_dir):
        ck = vrs._checkpoint_hyperparams(d)
        if ck is not None and ck.get("cql_alpha") is not None and \
                np.isclose(ck["cql_alpha"], target_alpha, rtol=1e-6, atol=1e-9):
            matches.append(d)
    if len(matches) != 1:
        raise ValueError(f"expected 1 run at cql_alpha={target_alpha}, found {matches}")
    return load_actor_and_critic(model_dir=matches[0], model_type="cql", state_dim=state_dim)


# =============================================================================
# 2. Saturation diagnostics
# =============================================================================

@torch.no_grad()
def actor_raw_params(actor, states, batch_size=4096):
    """Per-state gate probs, raw means and stds for both drugs."""
    actor.eval()
    dev = next(actor.parameters()).device
    states = _np(states)
    keys = ["f_p", "v_p", "f_raw_mean", "v_raw_mean", "f_std", "v_std"]
    out = {k: [] for k in keys}
    for i in range(0, len(states), batch_size):
        s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=dev)
        _, _, _, dist = actor(s, deterministic=True, return_dist=True)
        for k, t in zip(keys, dist):
            out[k].append(t.squeeze(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def saturation_summary(model_entry, processor, floor):
    actor, _, _, hp = model_entry
    p = actor_raw_params(actor, processor.rl_dataset["states"])
    clin_v = _np(processor.rl_dataset["actions"])[:, 1] > VASO_THRESHOLD
    vr, vp, vs = p["v_raw_mean"], p["v_p"], p["v_std"]
    det_dose = (vp > 0.5) * _sigmoid(vr)
    tri_w = np.clip(0.5 - np.abs(vp - 0.5), 0, None)
    return {
        "cql_alpha": hp.get("cql_alpha"),
        "vaso_coef": effective_vaso_coef(hp),
        "vaso_mean_floor": hp.get("vaso_mean_floor"),
        "v_raw_median": np.median(vr),
        "v_raw_p10": np.percentile(vr, 10),
        "v_raw_p90": np.percentile(vr, 90),
        "v_raw_sd_across_states": vr.std(),                 # ~0 => state-independent output
        "v_raw_median_clin_vaso": np.median(vr[clin_v]),
        "frac_below_floor_%": 100 * (vr < floor).mean(),
        "frac_saturated_%": 100 * (vr < SATURATED_RAW).mean(),
        "gate_p_mean": vp.mean(),
        "gate_p_sd_across_states": vp.std(),
        "gate_p_clin_vaso": vp[clin_v].mean(),
        "gate_auroc_clin_vaso": roc_auc_score(clin_v, vp) if 0 < clin_v.mean() < 1 else np.nan,
        "vaso_reg_weight_mean": tri_w.mean(),               # effective strength of existing regulariser
        "v_std_median": np.median(vs),
        "v_std_at_floor_%": 100 * np.isclose(vs, np.exp(LOG_STD_MIN), atol=1e-3).mean(),
        "v_std_at_ceiling_%": 100 * np.isclose(vs, np.exp(LOG_STD_MAX), atol=1e-3).mean(),
        "vaso_fire_%": 100 * (det_dose > VASO_THRESHOLD).mean(),
        "fluid_fire_%": 100 * (p["f_p"] > 0.5).mean(),
        "fluid_raw_median": np.median(p["f_raw_mean"]),
        "_v_raw_mean": vr,                                  # kept for plotting
    }


def saturation_table(models, processor, floor):
    """{label: model_entry} -> (DataFrame, {label: v_raw_mean array})."""
    rows, raws = {}, {}
    for label, entry in models.items():
        s = saturation_summary(entry, processor, floor)
        raws[label] = s.pop("_v_raw_mean")
        rows[label] = s
    return pd.DataFrame(rows).T, raws


# =============================================================================
# 3. Does the actor follow its own critic?
# =============================================================================

def gate_vs_critic(model_entry, processor):
    """
    At clinician-vaso states: own-critic dQ = Q(clin dose) - Q(no vaso), and
    how the actor's vaso outputs relate to it. A working actor should show
    higher gate p / raw mean where dQ > 0 (AUROC > 0.5, Spearman > 0).
    A saturated actor shows identical outputs in both groups.
    """
    actor, critic, _, hp = model_entry
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"])
    idx = np.where(clin[:, 1] > VASO_THRESHOLD)[0]
    scale = vrs._reward_scale(hp, "CQL")
    dq = vrs.vaso_delta_q(critic, states[idx], clin[idx]) / scale
    p = actor_raw_params(actor, states[idx])
    pos = dq > 0
    out = {
        "n_clin_vaso": len(idx),
        "frac_dq>0_%": 100 * pos.mean(),
        "dq_mean_HS": dq.mean(),
        "gate_p_median_dq>0": np.median(p["v_p"][pos]) if pos.any() else np.nan,
        "gate_p_median_dq<=0": np.median(p["v_p"][~pos]) if (~pos).any() else np.nan,
        "v_raw_median_dq>0": np.median(p["v_raw_mean"][pos]) if pos.any() else np.nan,
        "v_raw_median_dq<=0": np.median(p["v_raw_mean"][~pos]) if (~pos).any() else np.nan,
        "auroc_gate_p_vs_dq>0": roc_auc_score(pos, p["v_p"]) if 0 < pos.mean() < 1 else np.nan,
        "spearman_gate_p_dq": stats.spearmanr(p["v_p"], dq)[0],
        "spearman_v_raw_dq": stats.spearmanr(p["v_raw_mean"], dq)[0],
    }
    return out, {"dq": dq, "v_p": p["v_p"], "v_raw_mean": p["v_raw_mean"]}


def gate_vs_critic_table(models, processor):
    rows, per_state = {}, {}
    for label, entry in models.items():
        rows[label], per_state[label] = gate_vs_critic(entry, processor)
    return pd.DataFrame(rows).T, per_state


def plot_gate_vs_critic(per_state, labels=None, save_path=None):
    """Scatter of gate p and raw mean vs own-critic dQ at clinician-vaso states."""
    labels = labels or list(per_state)
    fig, axes = plt.subplots(2, len(labels), figsize=(3.6 * len(labels), 6.5), squeeze=False)
    for j, lab in enumerate(labels):
        d = per_state[lab]
        for i, (key, ylab) in enumerate([("v_p", "vaso gate p"), ("v_raw_mean", "vaso raw mean")]):
            ax = axes[i, j]
            ax.scatter(d["dq"], d[key], s=3, alpha=0.3, color="#2471a3")
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            if key == "v_raw_mean":
                ax.axhline(SATURATED_RAW, color="k", lw=0.8, ls=":")
            ax.set_xlabel("own-critic dQ (HS pts)")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(lab, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# 4. Before/after comparison using the vaso-sweep machinery
# =============================================================================

def evaluate_models(models, processor, hs, sev_edges, sarsa_models,
                    training_actions, training_health_score, **kwargs):
    """{label: model_entry} -> {label: [vrs.summarise_model result]} so
    vrs.behaviour_table / vrs.critic_table work with string labels."""
    results = {}
    for label, entry in models.items():
        print(f"=== {label} ===", flush=True)
        results[label] = [vrs.summarise_model(entry, processor, hs, sev_edges, sarsa_models,
                                              training_actions, training_health_score, **kwargs)]
    return results


# =============================================================================
# 5. Where does the SARSA advantage change come from? (fluid vs vaso)
# =============================================================================

def sarsa_advantage_decomposition(model_entry, processor, sarsa_models):
    """
    Splits each SARSA critic's advantage of the (deterministic) policy action
    over the clinician action into a fluid part and a vaso part:

        A_total = Q(s, a_pi) - Q(s, a_clin)
                = [Q(s, [f_pi, 0]) - Q(s, a_clin)]      <- "fluid part" (policy with vaso removed)
                + [Q(s, a_pi) - Q(s, [f_pi, 0])]        <- "vaso part"  (nonzero only where the policy fires)

    Normalised by each SARSA critic's Q IQR at clinician actions (as in the alpha
    sweep). Uses the mean of the twin heads, so absolute values can differ slightly
    from compute_model_advantage; the split is internally consistent.
    Caveat: the SARSA critics carry the confounded vaso penalty (severity-graded,
    -1.3 HS pts per unit dose at clinician-vaso states), so a negative vaso part is
    expected for ANY vaso-giving policy and is not independent evidence of harm.
    """
    from .evaluate_cql_alpha_sweep import get_critic

    actor = model_entry[0]
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"]).astype(np.float32)
    pol = vrs.actor_gate_outputs(actor, states)["actions"].astype(np.float32)
    pol_novaso = pol.copy()
    pol_novaso[:, 1] = 0.0
    fired = pol[:, 1] > VASO_THRESHOLD
    clin_v = clin[:, 1] > VASO_THRESHOLD

    rows = []
    for k, entry in enumerate(sarsa_models):
        critic = get_critic(entry)
        q_clin = vrs.critic_q(critic, states, clin)
        q_pol = vrs.critic_q(critic, states, pol)
        q_novaso = vrs.critic_q(critic, states, pol_novaso)
        iqr = np.subtract(*np.percentile(q_clin, [75, 25]))
        a_total = (q_pol - q_clin) / iqr
        a_fluid = (q_novaso - q_clin) / iqr
        a_vaso = (q_pol - q_novaso) / iqr
        rows.append({
            "sarsa_seed": k + 1,
            "adv_total": a_total.mean(),
            "adv_fluid_part": a_fluid.mean(),
            "adv_vaso_part": a_vaso.mean(),
            "vaso_part_per_fired_state": a_vaso[fired].mean() if fired.any() else np.nan,
            "vaso_part_fired_at_clin_vaso": a_vaso[fired & clin_v].mean() if (fired & clin_v).any() else np.nan,
            "vaso_part_fired_at_clin_zero": a_vaso[fired & ~clin_v].mean() if (fired & ~clin_v).any() else np.nan,
            "policy_fire_%": 100 * fired.mean(),
            "fired_at_clin_vaso_%_of_fires": 100 * (fired & clin_v).sum() / max(fired.sum(), 1),
        })
    df = pd.DataFrame(rows).set_index("sarsa_seed")
    df.loc["mean"] = df.mean()
    return df


def sarsa_advantage_decomposition_table(models, processor, sarsa_models):
    """{label: model_entry} -> one row per model (mean over SARSA seeds)."""
    return pd.DataFrame({label: sarsa_advantage_decomposition(entry, processor, sarsa_models).loc["mean"]
                         for label, entry in models.items()}).T


# =============================================================================
# 6. Is the floor setting the dose? (floor runs)
# =============================================================================

def fire_rate_by_dose_threshold(model_entry, processor, training_actions,
                                clinician_pcts=(0, 1, 5, 10, 25)):
    """
    Deterministic vaso fire rate when a 'fire' must reach a clinically meaningful
    dose: thresholds = percentiles of clinician nonzero doses (TRAIN). The 1e-4
    rate counts floor-level (~0.05) doses that clinicians essentially never give.
    """
    actor = model_entry[0]
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"])
    clin_v = clin[:, 1] > VASO_THRESHOLD
    pol_v = vrs.actor_gate_outputs(actor, states)["actions"][:, 1]
    tv = _np(training_actions)[:, 1]
    tv = tv[tv > VASO_THRESHOLD]
    rows = [{"threshold": "1e-4 (any)", "dose_threshold": VASO_THRESHOLD}]
    rows += [{"threshold": f"clinician p{p}", "dose_threshold": float(np.percentile(tv, p))} for p in clinician_pcts]
    for r in rows:
        f = pol_v > r["dose_threshold"]
        r.update({"fire_%": 100 * f.mean(),
                  "fire_%_clin_vaso": 100 * f[clin_v].mean(),
                  "fire_%_clin_zero": 100 * f[~clin_v].mean(),
                  "clinician_fire_%": 100 * (clin[:, 1] > r["dose_threshold"]).mean()})
    return pd.DataFrame(rows).set_index("threshold")


def q_dose_curve_at_fired(model_entry, processor, floor,
                          vaso_grid=(0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)):
    """
    Own-critic Q(s, [f_pi, v]) - Q(s, [f_pi, 0]) (HS points), fluid held at the
    POLICY's fluid, at states where the policy fires -- split by clinician vaso
    and by whether the actor's mean sits near the floor.
      - curve peaking at the smallest nonzero dose / falling with dose => the
        critic wants 'as little vaso as possible'; the floor sets the dose
      - near-zero gain at the floor dose => those fires are near-indifference
    Also returns the fraction of states whose critic-argmax dose is 0.
    """
    actor, critic, _, hp = model_entry
    states = _np(processor.rl_dataset["states"])
    clin_v = _np(processor.rl_dataset["actions"])[:, 1] > VASO_THRESHOLD
    g = vrs.actor_gate_outputs(actor, states)
    p = actor_raw_params(actor, states)
    pol = g["actions"].astype(np.float32)
    fired = pol[:, 1] > VASO_THRESHOLD
    near_floor = p["v_raw_mean"] < floor + 0.5
    scale = vrs._reward_scale(hp, "CQL")
    grid = np.asarray(vaso_grid, dtype=np.float32)

    groups = {"fired @ clin-vaso": fired & clin_v,
              "fired @ clin-zero": fired & ~clin_v,
              "fired, mean near floor": fired & near_floor,
              "fired, mean well above floor": fired & ~near_floor,
              "not fired (all)": ~fired}
    rows = {}
    for name, m in groups.items():
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        curves = vrs.q_vaso_dose_curves(critic, states[idx], pol[idx], grid) / scale
        curves = curves - curves[:, [0]]
        row = {"n": len(idx)}
        row.update({f"v={v:g}": c for v, c in zip(grid, curves.mean(0))})
        row["argmax_dose_is_0_%"] = 100 * (curves.argmax(1) == 0).mean()
        row["argmax_dose_median"] = float(np.median(grid[curves.argmax(1)]))
        rows[name] = row
    return pd.DataFrame(rows).T


# =============================================================================
# 7. Does the actor's dose follow its critic's preferred dose? (floor runs)
# =============================================================================

def _interp_rows(x, grid, curves):
    """Row-wise linear interpolation of curves (n, G) at points x (n,)."""
    x = np.clip(x, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2)
    w = (x - grid[j]) / (grid[j + 1] - grid[j])
    rows = np.arange(len(x))
    y0, y1 = curves[rows, j], curves[rows, j + 1]
    return y0 + w * (y1 - y0), (y1 - y0) / (grid[j + 1] - grid[j])


def actor_vs_critic_dose(model_entry, processor, floor,
                         grid=np.round(np.arange(0.0, 0.605, 0.01), 3)):
    """
    At states where the policy fires: the actor's deterministic dose vs its own
    critic's preferred dose (argmax of Q(s, [f_pi, v]) over a fine vaso grid,
    fluid held at the policy's). Per state it also returns
      - regret: Q(best dose) - Q(actor dose), HS points (value left on the table)
      - slope:  dQ/dv at the actor's dose, HS points per unit dose
                (> 0 => critic wants MORE vaso than the actor gives)
    Returns (summary DataFrame by group, per-state DataFrame).
    """
    actor, critic, _, hp = model_entry
    states = _np(processor.rl_dataset["states"])
    clin_v = _np(processor.rl_dataset["actions"])[:, 1] > VASO_THRESHOLD
    pol = vrs.actor_gate_outputs(actor, states)["actions"].astype(np.float32)
    raw = actor_raw_params(actor, states)["v_raw_mean"]
    idx = np.where(pol[:, 1] > VASO_THRESHOLD)[0]
    grid = np.asarray(grid, dtype=np.float64)
    scale = vrs._reward_scale(hp, "CQL")

    curves = vrs.q_vaso_dose_curves(critic, states[idx], pol[idx], grid) / scale
    curves = curves - curves[:, [0]]
    best = grid[curves.argmax(1)]
    actor_dose = pol[idx, 1].astype(np.float64)
    q_actor, slope = _interp_rows(actor_dose, grid, curves)

    per_state = pd.DataFrame({
        "state_idx": idx,
        "clin_vaso": clin_v[idx],
        "near_floor": raw[idx] < floor + 0.5,
        "actor_dose": actor_dose,
        "critic_best_dose": best,
        "q_gain_actor": q_actor,
        "q_gain_best": curves.max(1),
        "regret": curves.max(1) - q_actor,
        "slope_at_actor_dose": slope,
    })

    def summarise(d):
        pos = d["critic_best_dose"] > 0
        return pd.Series({
            "n": len(d),
            "critic_best_is_0_%": 100 * (~pos).mean(),
            "spearman_actor_vs_best": stats.spearmanr(d["actor_dose"], d["critic_best_dose"])[0] if len(d) > 10 else np.nan,
            "spearman_actor_vs_best (best>0)": (stats.spearmanr(d.loc[pos, "actor_dose"], d.loc[pos, "critic_best_dose"])[0]
                                                if pos.sum() > 10 else np.nan),
            "actor_dose_median": d["actor_dose"].median(),
            "critic_best_median": d["critic_best_dose"].median(),
            "actor_below_best_%": 100 * (d["actor_dose"] < d["critic_best_dose"]).mean(),
            "median_ratio_actor/best (best>0)": (d.loc[pos, "actor_dose"] / d.loc[pos, "critic_best_dose"]).median(),
            "regret_median_HS": d["regret"].median(),
            "regret_mean_HS": d["regret"].mean(),
            "q_gain_best_median_HS": d["q_gain_best"].median(),
            "slope_at_actor_median": d["slope_at_actor_dose"].median(),
            "slope>0_%": 100 * (d["slope_at_actor_dose"] > 0).mean(),
        })

    groups = {"all fired": per_state,
              "fired @ clin-vaso": per_state[per_state.clin_vaso],
              "fired @ clin-zero": per_state[~per_state.clin_vaso],
              "mean near floor": per_state[per_state.near_floor],
              "mean well above floor": per_state[~per_state.near_floor]}
    summary = pd.DataFrame({k: summarise(v) for k, v in groups.items() if len(v)}).T
    return summary, per_state


def plot_actor_vs_critic_dose(per_state, floor, save_path=None):
    """2D histograms of actor dose vs critic-preferred dose (identity line and
    floor dose marked), plus the regret distribution, by clinician vaso."""
    floor_dose = _sigmoid(floor)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    hi = max(per_state["actor_dose"].max(), per_state["critic_best_dose"].max()) * 1.02
    bins = np.linspace(0, hi, 40)
    for ax, (name, m) in zip(axes[:2], [("clinician gave vaso", per_state.clin_vaso),
                                        ("clinician gave none", ~per_state.clin_vaso)]):
        d = per_state[m]
        h = ax.hist2d(d["critic_best_dose"], d["actor_dose"], bins=[bins, bins], cmap="Blues", cmin=1)
        fig.colorbar(h[3], ax=ax, label="states")
        ax.plot([0, hi], [0, hi], color="k", lw=0.8, ls="--", label="actor = critic")
        ax.axhline(floor_dose, color="#c0392b", lw=0.8, ls=":", label=f"floor dose {floor_dose:.3f}")
        ax.set_xlabel("critic-preferred dose (argmax Q)")
        ax.set_ylabel("actor dose (deterministic)")
        ax.set_title(f"Fired @ {name} (n={len(d)})", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[2]
    rb = np.linspace(0, np.percentile(per_state["regret"], 99), 50)
    for name, m, c in [("clinician gave vaso", per_state.clin_vaso, "#c0392b"),
                       ("clinician gave none", ~per_state.clin_vaso, "#2471a3")]:
        ax.hist(per_state.loc[m, "regret"], bins=rb, histtype="step", lw=1.5, color=c, density=True, label=name)
    ax.set_xlabel("regret: Q(best) - Q(actor dose)  (HS points)")
    ax.set_ylabel("density")
    ax.set_title("Value left on the table", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# 8. Multi-model (multi-seed) wrappers for sections 6-7
# =============================================================================

def floor_binding(model_entry, processor, floor):
    """Where the actor fires (gate p > 0.5): how often its vaso mean sits at/below
    the floor, and the dose distribution, split by clinician vaso."""
    clin_v = _np(processor.rl_dataset["actions"])[:, 1] > VASO_THRESHOLD
    p = actor_raw_params(model_entry[0], processor.rl_dataset["states"])
    fired = p["v_p"] > 0.5
    vr = p["v_raw_mean"]
    rows = {}
    for name, m in [("all fired", fired), ("fired @ clin-vaso", fired & clin_v), ("fired @ clin-zero", fired & ~clin_v)]:
        if not m.any():
            rows[name] = {"n": 0}
            continue
        d10, d50, d90 = _sigmoid(np.percentile(vr[m], [10, 50, 90]))
        rows[name] = {"n": int(m.sum()),
                      "below_floor_%": 100 * (vr[m] < floor).mean(),
                      "within_0.5_of_floor_%": 100 * (vr[m] < floor + 0.5).mean(),
                      "dose_p10": d10, "dose_p50": d50, "dose_p90": d90}
    return pd.DataFrame(rows).T


def _stack(per_model):
    """{label: DataFrame} -> one DataFrame indexed (model, row)."""
    return pd.concat(per_model, names=["model", "group"])


def seed_aggregate(stacked):
    """Mean / min / max over models for each group (row) of a stacked table."""
    num = stacked.apply(pd.to_numeric, errors="coerce")
    return num.groupby(level=1, sort=False).agg(["mean", "min", "max"])


def floor_binding_table(models, processor, floor):
    return _stack({k: floor_binding(m, processor, floor) for k, m in models.items()})


def fire_rate_by_dose_threshold_table(models, processor, training_actions, **kwargs):
    return _stack({k: fire_rate_by_dose_threshold(m, processor, training_actions, **kwargs)
                   for k, m in models.items()})


def q_dose_curve_at_fired_table(models, processor, floor, **kwargs):
    out = {}
    for k, m in models.items():
        print(f"  Q-dose curves: {k}", flush=True)
        out[k] = q_dose_curve_at_fired(m, processor, floor, **kwargs)
    return _stack(out)


def actor_vs_critic_dose_table(models, processor, floor, **kwargs):
    """Returns (stacked summary, {label: per-state DataFrame})."""
    summaries, per_state = {}, {}
    for k, m in models.items():
        print(f"  actor vs critic dose: {k}", flush=True)
        summaries[k], per_state[k] = actor_vs_critic_dose(m, processor, floor, **kwargs)
    return _stack(summaries), per_state


def clinically_meaningful_rates(per_state, processor, training_actions, pcts=(5, 10)):
    """
    Fire rate at clinically meaningful doses (clinician TRAIN percentiles):
    the actor's actual dose vs the dose its own critic prefers (upper bound),
    as % of ALL states, alongside the clinicians' rate at the same threshold.
    """
    tv = _np(training_actions)[:, 1]
    tv = tv[tv > VASO_THRESHOLD]
    clin = _np(processor.rl_dataset["actions"])[:, 1]
    n_states = len(clin)
    rows = []
    for p in pcts:
        thr = float(np.percentile(tv, p))
        r = {"threshold": f"clinician p{p}", "dose_threshold": thr,
             "actor_fire_%_of_states": 100 * (per_state.actor_dose >= thr).sum() / n_states,
             "critic_best_fire_%_of_states": 100 * (per_state.critic_best_dose >= thr).sum() / n_states,
             "clinician_fire_%": 100 * (clin > thr).mean()}
        for name, m in [("clin-vaso", per_state.clin_vaso), ("clin-zero", ~per_state.clin_vaso)]:
            d = per_state[m]
            r[f"fired@{name}: actor>=thr_%"] = 100 * (d.actor_dose >= thr).mean() if len(d) else np.nan
            r[f"fired@{name}: critic_best>=thr_%"] = 100 * (d.critic_best_dose >= thr).mean() if len(d) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("threshold")


def clinically_meaningful_rates_table(per_state_by_model, processor, training_actions, **kwargs):
    return _stack({k: clinically_meaningful_rates(ps, processor, training_actions, **kwargs)
                   for k, ps in per_state_by_model.items()})


# =============================================================================
# 9. Gate decisiveness and seed agreement (floor runs)
# =============================================================================

def gate_decisiveness(model_entry, processor):
    """
    How decisive is the vaso gate where it fires? Many fired states with p just
    above 0.5 => the deterministic fire/no-fire split is set by near-indifference
    (tiny critic differences + gate entropy bonus), not by a clear critic preference.
    """
    hp = model_entry[3]
    p = actor_raw_params(model_entry[0], processor.rl_dataset["states"])["v_p"]
    fired = p > 0.5
    return {
        "gate_entropy_scale": hp.get("actor_loss_gate_entropy_scale"),
        "fire_%": 100 * fired.mean(),
        "gate_p_median_when_fired": np.median(p[fired]) if fired.any() else np.nan,
        "fired_with_p<0.6_%": 100 * (p[fired] < 0.6).mean() if fired.any() else np.nan,
        "fired_with_p>0.9_%": 100 * (p[fired] > 0.9).mean() if fired.any() else np.nan,
        "all_states_p_in_[0.4,0.6]_%": 100 * ((p > 0.4) & (p < 0.6)).mean(),
        "all_states_p<0.1_%": 100 * (p < 0.1).mean(),
    }


def gate_decisiveness_table(models, processor):
    return pd.DataFrame({k: gate_decisiveness(m, processor) for k, m in models.items()}).T


def seed_agreement(models, processor, dose_threshold=VASO_THRESHOLD):
    """
    Per state: how many models (seeds) give vaso above dose_threshold.
    Returns (summary DataFrame by number of agreeing seeds, per-state counts).
    States where most seeds agree are the policy's robust vaso recommendations;
    states where only one or two seeds fire are seed-dependent.
    """
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"])[:, 1]
    clin_v = clin > VASO_THRESHOLD
    fires = np.stack([vrs.actor_gate_outputs(m[0], states)["actions"][:, 1] > dose_threshold
                      for m in models.values()])
    count = fires.sum(0)
    n_models = fires.shape[0]
    rows = []
    for k in range(n_models + 1):
        m = count == k
        rows.append({"n_seeds_firing": k, "n_states": int(m.sum()), "%_of_states": 100 * m.mean(),
                     "clin_vaso_%_within": 100 * clin_v[m].mean() if m.any() else np.nan})
    df = pd.DataFrame(rows).set_index("n_seeds_firing")
    pairwise = [np.mean(fires[i] & fires[j]) / max(np.mean(fires[i] | fires[j]), 1e-12)
                for i in range(n_models) for j in range(i + 1, n_models)]
    print(f"dose threshold {dose_threshold:.3g}: mean pairwise Jaccard of fired states across seeds = "
          f"{np.mean(pairwise):.3f} (range {np.min(pairwise):.3f}-{np.max(pairwise):.3f}); "
          f"clinician vaso prevalence {100 * clin_v.mean():.1f}%")
    return df, count


# =============================================================================
# 10. Do the seeds' CRITICS agree about vaso? And are the gates saturated?
# =============================================================================

def critic_vaso_gain(model_entry, processor, dose=0.1):
    """Own-critic Q(s, [f_clin, dose]) - Q(s, [f_clin, 0]) at every state (HS points).
    Clinician fluid is used so every model is evaluated on the same action pair."""
    _, critic, _, hp = model_entry
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"]).astype(np.float32)
    a1, a0 = clin.copy(), clin.copy()
    a1[:, 1], a0[:, 1] = dose, 0.0
    scale = vrs._reward_scale(hp, "CQL")
    return (vrs.critic_q(critic, states, a1) - vrs.critic_q(critic, states, a0)) / scale


def critic_agreement(models, processor, dose=0.1):
    """
    Across seeds: do the critics agree on WHERE vaso is least penalised / helpful,
    and do the gates follow their own critic vs the other seeds' critics?
      - pairwise Spearman of critic gain across states (all / clinician-vaso)
      - sign agreement: % of states where all critics agree on the sign of the gain
      - per model: Spearman(gate logit, own critic gain) vs mean Spearman with the
        OTHER seeds' critics; and gate saturation at fired states the own critic
        disfavours (gain < 0 but gate p > 0.99)
    Critics agree but gates don't => actor-side problem (e.g. gate saturation).
    Critics disagree => the vaso signal itself is not reproducible across seeds.
    """
    labels = list(models)
    clin_v = _np(processor.rl_dataset["actions"])[:, 1] > VASO_THRESHOLD
    gains = np.stack([critic_vaso_gain(models[k], processor, dose) for k in labels])
    gate_p = np.stack([actor_raw_params(models[k][0], processor.rl_dataset["states"])["v_p"] for k in labels])
    gate_logit = _logit(gate_p, eps=1e-6)

    def pairwise(mask):
        n = len(labels)
        M = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = stats.spearmanr(gains[i, mask], gains[j, mask])[0]
        return pd.DataFrame(M, index=labels, columns=labels)

    all_mask = np.ones(gains.shape[1], bool)
    rho_all, rho_cv = pairwise(all_mask), pairwise(clin_v)
    off = ~np.eye(len(labels), dtype=bool)
    signs = np.sign(gains)
    print(f"critic gain at dose {dose}: mean pairwise Spearman across seeds = "
          f"{rho_all.values[off].mean():.3f} (all states), {rho_cv.values[off].mean():.3f} (clinician-vaso states)")
    print(f"  all critics agree on sign of gain at {100 * (np.abs(signs.sum(0)) == len(labels)).mean():.1f}% of states; "
          f"gain > 0 in ALL critics at {100 * (gains > 0).all(0).mean():.2f}% of states")

    rows = {}
    for i, k in enumerate(labels):
        fired = gate_p[i] > 0.5
        others = [stats.spearmanr(gate_logit[i], gains[j])[0] for j in range(len(labels)) if j != i]
        disfav = fired & (gains[i] < 0)
        rows[k] = {
            "spearman_gate_vs_own_critic": stats.spearmanr(gate_logit[i], gains[i])[0],
            "spearman_gate_vs_other_critics_mean": np.mean(others),
            "own_critic_gain>0_%": 100 * (gains[i] > 0).mean(),
            "fired_where_own_critic_gain<0_%": 100 * disfav.sum() / max(fired.sum(), 1),
            "of_those_gate_p>0.99_%": 100 * (gate_p[i][disfav] > 0.99).mean() if disfav.any() else np.nan,
            "not_fired_where_own_critic_gain>0_%": 100 * (~fired & (gains[i] > 0)).sum() / max((gains[i] > 0).sum(), 1),
        }
    return {"rho_all": rho_all, "rho_clin_vaso": rho_cv, "per_model": pd.DataFrame(rows).T, "gains": gains}


# =============================================================================
# 11. Generalised critic agreement: SARSA critics, and fluid as well as vaso
# =============================================================================

DRUG_IDX = {"fluid": 0, "vaso": 1}


def cql_critics(models):
    """{label: CQL model_entry} -> {label: (critic, reward_scale)}"""
    return {k: (m[1], vrs._reward_scale(m[3], k)) for k, m in models.items()}


def sarsa_critics(sarsa_models):
    """[(critic, cm, hp), ...] -> {'SARSA k': (critic, reward_scale)}"""
    from .evaluate_cql_alpha_sweep import get_critic
    out = {}
    for k, entry in enumerate(sarsa_models):
        hp = entry[2] if len(entry) >= 3 else {}
        out[f"SARSA {k + 1}"] = (get_critic(entry), vrs._reward_scale(hp, f"SARSA {k + 1}"))
    return out


def critic_gain(critic, scale, states, clin, drug="vaso", mode="absolute", dose=0.1, delta=0.1):
    """
    Per-state critic gain for one drug, the other drug held at the clinician's dose (HS points).
      mode="absolute": Q(dose) - Q(0)                   -- is giving the drug better than none?
      mode="local":    Q(d_clin + delta) - Q(d_clin - delta), doses clipped to [0, 1]
                                                         -- would the critic give more or less
                                                            than the clinician did? (fluid is
                                                            given at ~91% of states, so this is
                                                            the more relevant fluid question)
    """
    j = DRUG_IDX[drug]
    clin = clin.astype(np.float32)
    hi, lo = clin.copy(), clin.copy()
    if mode == "absolute":
        hi[:, j], lo[:, j] = dose, 0.0
    elif mode == "local":
        hi[:, j] = np.clip(clin[:, j] + delta, 0, 1)
        lo[:, j] = np.clip(clin[:, j] - delta, 0, 1)
    else:
        raise ValueError(mode)
    return (vrs.critic_q(critic, states, hi) - vrs.critic_q(critic, states, lo)) / scale


def critic_gain_agreement(critics, processor, drug="vaso", mode="absolute", dose=None, delta=0.1):
    """
    Agreement across a set of critics (e.g. the 5 CQL floor seeds, or the 5 SARSA seeds)
    on a per-state gain for one drug.
      - pairwise Spearman across states (all states / states where the clinician gave the drug)
      - sign agreement, and what INDEPENDENT critics with the same base rates would give
      - ensemble view: % states where the mean gain > 0, where all > 0, and where the
        lower 95% bound across critics (mean - 1.96 SD/sqrt(n)) > 0
    dose=None -> vaso 0.1; fluid = median clinician nonzero fluid dose.
    """
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"])
    j = DRUG_IDX[drug]
    if dose is None:
        nz = clin[clin[:, j] > VASO_THRESHOLD, j]
        dose = 0.1 if drug == "vaso" else float(np.median(nz))
    labels = list(critics)
    gains = np.stack([critic_gain(c, s, states, clin, drug, mode, dose, delta) for c, s in critics.values()])
    clin_given = clin[:, j] > VASO_THRESHOLD

    def pairwise(mask):
        n = len(labels)
        M = np.eye(n)
        for a in range(n):
            for b in range(a + 1, n):
                M[a, b] = M[b, a] = stats.spearmanr(gains[a, mask], gains[b, mask])[0]
        return pd.DataFrame(M, index=labels, columns=labels)

    rho_all = pairwise(np.ones(gains.shape[1], bool))
    rho_given = pairwise(clin_given)
    off = ~np.eye(len(labels), dtype=bool)

    pos = gains > 0
    p_pos = pos.mean(1)
    observed_sign_agree = (pos.all(0) | (~pos).all(0)).mean()
    expected_sign_agree = np.prod(p_pos) + np.prod(1 - p_pos)
    mean_g, sd_g = gains.mean(0), gains.std(0, ddof=1)
    lcb = mean_g - 1.96 * sd_g / np.sqrt(len(labels))

    desc = f"{drug}, {mode}" + (f", dose={dose:.3g}" if mode == "absolute" else f", delta={delta}")
    summary = pd.Series({
        "setting": desc,
        "n_critics": len(labels),
        "mean_pairwise_spearman_all": rho_all.values[off].mean(),
        f"mean_pairwise_spearman_clin_{drug}_given": rho_given.values[off].mean(),
        "gain>0_%_per_critic (mean)": 100 * p_pos.mean(),
        "gain>0_%_per_critic (range)": f"{100 * p_pos.min():.1f}-{100 * p_pos.max():.1f}",
        "sign_agreement_observed_%": 100 * observed_sign_agree,
        "sign_agreement_if_independent_%": 100 * expected_sign_agree,
        "ensemble_mean_gain>0_%": 100 * (mean_g > 0).mean(),
        "all_critics_gain>0_%": 100 * pos.all(0).mean(),
        "ensemble_lower_bound>0_%": 100 * (lcb > 0).mean(),
        "ensemble_mean_gain_median_HS": float(np.median(mean_g)),
    })
    return {"summary": summary, "rho_all": rho_all, f"rho_clin_{drug}_given": rho_given, "gains": gains}


def critic_agreement_table(results):
    """{name: critic_gain_agreement output} -> one summary table."""
    return pd.DataFrame({k: v["summary"] for k, v in results.items()}).T


# =============================================================================
# 12. Is the critic disagreement extrapolation? (in-distribution vs OOD Q values)
# =============================================================================

def q_value_agreement(critics, processor, ood_vaso_dose=0.1):
    """
    Across a set of critics: pairwise Spearman of Q VALUES (not action gains) on
      - in-distribution actions:  Q(s, a_clin) at all states, and at clinician-vaso states
      - out-of-distribution:      Q(s, [f_clin, ood_vaso_dose]) at clinician-ZERO states
                                  (vaso where clinicians gave none)
    High agreement in-distribution but low OOD => the disagreement is extrapolation
    error on actions the data don't cover (which only enter CQL's Bellman targets and
    conservative penalty once the actor starts proposing them).
    Low agreement in-distribution too => the critics differ more broadly.
    """
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"]).astype(np.float32)
    clin_v = clin[:, 1] > VASO_THRESHOLD
    a_ood = clin.copy()
    a_ood[:, 1] = ood_vaso_dose

    q_in = np.stack([vrs.critic_q(c, states, clin) / s for c, s in critics.values()])
    q_ood = np.stack([vrs.critic_q(c, states, a_ood) / s for c, s in critics.values()])

    def mean_pairwise(Q, mask):
        n = Q.shape[0]
        vals = [stats.spearmanr(Q[i, mask], Q[j, mask])[0] for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

    out = {}
    for name, Q, mask in [("Q(s,a_clin) all states", q_in, np.ones(len(clin), bool)),
                          ("Q(s,a_clin) clin-vaso states", q_in, clin_v),
                          ("Q(s,a_clin) clin-zero states", q_in, ~clin_v),
                          (f"Q(s,[f_clin,v={ood_vaso_dose}]) clin-zero states (OOD)", q_ood, ~clin_v)]:
        m, lo, hi = mean_pairwise(Q, mask)
        out[f"{name}: mean"] = m
        out[f"{name}: range"] = f"{lo:.2f}-{hi:.2f}"
    return pd.Series(out)


def q_value_agreement_table(critic_sets, processor, **kwargs):
    rows = {}
    for name, critics in critic_sets.items():
        print(f"  Q-value agreement: {name}", flush=True)
        rows[name] = q_value_agreement(critics, processor, **kwargs)
    return pd.DataFrame(rows).T


# =============================================================================
# 13. Critic sanity: state values vs action effects
# =============================================================================

def critic_sanity(critic_sets, processor, health_score, sarsa_key="SARSA", fixed_action=(0.5, 0.0)):
    """
    Per critic, separates how much of Q(s, a_clin) is STATE value vs ACTION effect:
      - Q_fixed(s) = Q(s, fixed_action): a common action at every state -> a state-value proxy
      - action part = Q(s, a_clin) - Q_fixed(s)
    Reports, per critic:
      - Spearman(Q_fixed, health score)            -- does the critic value healthier states more?
      - Spearman(Q_fixed, SARSA-ensemble Q_fixed)   -- does it agree with the bonus-free reference?
      - action_share = var(action part) / var(Q(s, a_clin))
      - Q(s, a_clin) mean and IQR (HS points)
    and, per set, the mean pairwise Spearman of Q_fixed (state-value agreement across seeds).
    If critics agree on Q_fixed but not on Q(s, a_clin), their disagreement is in the ACTION
    effects (e.g. fluid), not in how they value states.
    """
    states = _np(processor.rl_dataset["states"])
    clin = _np(processor.rl_dataset["actions"]).astype(np.float32)
    a_fix = np.tile(np.asarray(fixed_action, dtype=np.float32), (len(clin), 1))
    hs = np.asarray(health_score)

    q_fix, q_clin = {}, {}
    for set_name, critics in critic_sets.items():
        for lab, (c, s) in critics.items():
            key = (set_name, lab)
            q_fix[key] = vrs.critic_q(c, states, a_fix) / s
            q_clin[key] = vrs.critic_q(c, states, clin) / s

    sarsa_ref = None
    sarsa_keys = [k for k in q_fix if k[0] == sarsa_key]
    if sarsa_keys:
        sarsa_ref = np.mean([q_fix[k] for k in sarsa_keys], axis=0)

    rows = {}
    for key in q_fix:
        qa, qf = q_clin[key], q_fix[key]
        act = qa - qf
        rows[key] = {
            "spearman_Qfixed_vs_health_score": stats.spearmanr(qf, hs)[0],
            "spearman_Qfixed_vs_SARSA": stats.spearmanr(qf, sarsa_ref)[0] if sarsa_ref is not None else np.nan,
            "action_share_of_Q_variance": act.var() / qa.var(),
            "Q_clin_mean_HS": qa.mean(),
            "Q_clin_IQR_HS": np.subtract(*np.percentile(qa, [75, 25])),
            "Q_fixed_IQR_HS": np.subtract(*np.percentile(qf, [75, 25])),
        }
    per_critic = pd.DataFrame(rows).T
    per_critic.index = pd.MultiIndex.from_tuples(per_critic.index, names=["set", "critic"])

    agree = {}
    for set_name in critic_sets:
        keys = [k for k in q_fix if k[0] == set_name]
        vals = [stats.spearmanr(q_fix[a], q_fix[b])[0] for i, a in enumerate(keys) for b in keys[i + 1:]]
        if not vals:   # a set with a single critic has no pairs
            agree[set_name] = {"Q_fixed_pairwise_spearman_mean": np.nan, "Q_fixed_pairwise_spearman_range": "n/a (1 critic)"}
            continue
        agree[set_name] = {"Q_fixed_pairwise_spearman_mean": np.mean(vals),
                           "Q_fixed_pairwise_spearman_range": f"{np.min(vals):.2f}-{np.max(vals):.2f}"}
    return per_critic, pd.DataFrame(agree).T


# =============================================================================
# 14. Dynamic DeepHit seeds: matched processors, DDH comparison, and which DDH
#     seed a model was trained on (Bellman residual)
# =============================================================================
#
# The data processor (data_processor_for_shards.py) picks the DDH output like this:
#   seed_mode == 'single_seed' -> the one 'unknown*'/'local*' folder in state_path/reward_path
#   otherwise                  -> the folder whose results.json hyperparameters.seed
#                                 == config_manager.training_config['seed']
# Both STATES (model_outputs) and REWARDS (health score) come from that DDH output,
# so RL models trained on different DDH seeds see different state embeddings AND
# different health-score rewards. Embeddings from different DDH runs are different
# coordinate systems: a model must be evaluated on the embedding it was trained on.

def list_ddh_seed_folders(ddh_dir):
    """{ddh_seed: folder path} for the seed folders inside one DDH results directory."""
    out = {}
    for f in sorted(os.listdir(ddh_dir)):
        if not (f.startswith("unknown") or f.startswith("local")):
            continue
        rj = os.path.join(ddh_dir, f, "results.json")
        seed = None
        if os.path.isfile(rj):
            with open(rj) as fh:
                seed = json.load(fh).get("hyperparameters", {}).get("seed")
        out[seed] = os.path.join(ddh_dir, f)
    return out


def load_processors_for_ddh(base_config_manager, base_hyperparams, ddh_dir, seed_mode, ddh_seed=None):
    """
    Processors built on a specific DDH output. Copies the base config manager,
    points state_path/reward_path at ddh_dir, and (for multi-seed DDH dirs) sets
    training_config['seed'] = ddh_seed so the processor picks that DDH seed folder.
    seed_mode: 'single_seed' or the multi-seed value your five-seed runs use
               (read it from one of their config managers: m[2].seed_mode).
    """
    import copy as _copy
    from src.rl import evaluate_terminal_reward_models as _etm
    cm = _copy.deepcopy(base_config_manager)
    cm.data_config["state_path"] = ddh_dir
    cm.data_config["reward_path"] = ddh_dir
    cm.seed_mode = seed_mode
    if ddh_seed is not None:
        cm.training_config["seed"] = ddh_seed
    hp = dict(base_hyperparams)
    hp["add_reward_to_vaso_dosing"] = False
    procs = {}
    procs["train"], procs["val"], procs["test"] = _etm.load_data(cm, hp)
    return procs


def check_row_alignment(processors_by_key, split="val"):
    """Do the processors line up row-for-row (same stays, same timesteps)? Required
    before comparing models trained on different DDH seeds state-by-state."""
    keys = list(processors_by_key)
    ref = processors_by_key[keys[0]][split]
    ref_ptr = np.asarray(ref.dynamic_deephit_predictions["index_ptr"])
    ref_act = _np(ref.rl_dataset["actions"])
    rows = {}
    for k in keys:
        p = processors_by_key[k][split]
        ptr = np.asarray(p.dynamic_deephit_predictions["index_ptr"])
        act = _np(p.rl_dataset["actions"])
        rows[k] = {"n_states": len(act),
                   "same_index_ptr_as_first": ptr.shape == ref_ptr.shape and np.array_equal(ptr, ref_ptr),
                   "same_actions_as_first": act.shape == ref_act.shape and np.allclose(act, ref_act)}
    return pd.DataFrame(rows).T


def compare_ddh_seeds(processors_by_key, split="val"):
    """
    Across DDH seeds (same rows required): per-timestep agreement of the health score
    and of its hourly change -- i.e. how different the REWARD is between DDH seeds.
    Latent embeddings are not compared (not comparable across separately trained models).
    """
    keys = list(processors_by_key)
    hs, dhs = {}, {}
    for k in keys:
        hs[k], dhs[k] = per_state_health_score(processors_by_key[k][split])
    n = len(keys)
    rho_hs, rho_dhs = np.eye(n), np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho_hs[i, j] = rho_hs[j, i] = stats.spearmanr(hs[keys[i]], hs[keys[j]])[0]
            rho_dhs[i, j] = rho_dhs[j, i] = stats.spearmanr(dhs[keys[i]], dhs[keys[j]])[0]
    summary = pd.DataFrame({k: {"hs_mean": hs[k].mean(), "hs_sd": hs[k].std(),
                                "dhs_sd": dhs[k].std()} for k in keys}).T
    return {"summary": summary,
            "rho_health_score": pd.DataFrame(rho_hs, index=keys, columns=keys),
            "rho_hourly_change": pd.DataFrame(rho_dhs, index=keys, columns=keys)}


def _next_rows(processor):
    """Index of the next state within the same trajectory (-1 at trajectory ends)."""
    dones = np.asarray(processor.rl_dataset["dones"]).astype(bool)
    nxt = np.arange(len(dones)) + 1
    nxt[dones] = -1
    nxt[nxt >= len(dones)] = -1
    return nxt


def bellman_residual(critic, processor, gamma, batch_size=4096):
    """
    SARSA-style residual using the logged data: |Q(s,a) - (r + gamma * Q(s',a'))|,
    with a' = the clinician's next action, terminal rows using r only.
    A critic evaluated on the states it was trained on should have a much smaller
    residual than on another DDH seed's states -- used to identify a model's DDH seed.
    Reward is taken from processor.rl_dataset['rewards'] (same reward settings as the
    processor's config; compare candidates RELATIVE to each other).
    """
    states = _np(processor.rl_dataset["states"])
    actions = _np(processor.rl_dataset["actions"]).astype(np.float32)
    rewards = _np(processor.rl_dataset["rewards"]).reshape(-1)
    nxt = _next_rows(processor)
    q = vrs.critic_q(critic, states, actions, batch_size)
    has_next = nxt >= 0
    q_next = np.zeros_like(q)
    q_next[has_next] = q[nxt[has_next]]
    target = rewards + gamma * q_next * has_next
    res = np.abs(q - target)
    return {"median_abs_residual": float(np.median(res)), "mean_abs_residual": float(res.mean()),
            "Q_sd": float(q.std())}


def identify_ddh_seed(models, processors_by_key, split="val", gamma=None):
    """
    For each model, Bellman residual of its critic on each candidate DDH processor.
    The matched DDH seed is the column with the smallest residual (relative to Q_sd).
    Returns a table of median |residual| / Q_sd (lower = better match) and the best match.
    """
    rows = {}
    for label, m in models.items():
        hp = m[3]
        g = gamma if gamma is not None else hp.get("discount", 0.975)
        r = {}
        for key, procs in processors_by_key.items():
            br = bellman_residual(m[1], procs[split], g)
            r[key] = br["median_abs_residual"] / max(br["Q_sd"], 1e-8)
        r["best_match"] = min((k for k in processors_by_key), key=lambda k: r[k])
        rows[label] = r
    return pd.DataFrame(rows).T