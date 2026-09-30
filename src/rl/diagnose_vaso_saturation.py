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