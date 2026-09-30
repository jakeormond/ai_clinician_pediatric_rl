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

def load_models(model_dir, state_dim, label_fmt="seed {seed}"):
    """Loads every run under model_dir -> {label: model_entry}."""
    out = {}
    for d in get_model_dirs(model_dir):
        entry = load_actor_and_critic(model_dir=d, model_type="cql", state_dim=state_dim)
        hp = entry[3]
        label = label_fmt.format(seed=hp.get("seed"), alpha=hp.get("cql_alpha"),
                                 coef=hp.get("vaso_reward_coefficient"))
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
        "vaso_coef": hp.get("vaso_reward_coefficient", 0.0) if hp.get("add_reward_to_vaso_dosing") else 0.0,
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