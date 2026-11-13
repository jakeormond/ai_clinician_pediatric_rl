"""
src/rl/evaluate_bc_final_seeds.py

Evaluation of the final behaviour-cloning (BC) seed models, written against the generic analysis API
(src/rl/analysis: runs.py, models.py, data.py). The BC actor is the behaviour density pi_b in WIS, so what matters
is whether it is a good DENSITY of the logged actions, not whether its point predictions are close:

  - log-likelihood against a no-state baseline (constant gate rate + one Gaussian over logit dose)
  - gate calibration (does a predicted fire probability of 0.3 mean 30% of the time?), per drug
  - magnitude calibration (are the logged logit-doses ~N(0,1) in units of BC's own mean and std?)
  - all of the above by severity (the sickest hours are where vaso is given and WIS weights are made)
  - agreement across seeds, and how much of a stay's gate log-likelihood comes from zero-fluid hours

Conventions
  * models: {seed: LoadedModel} (family 'bc'); procs: {seed: processor}, each seed on ITS OWN states
    (store.processor(model, split)); every function takes the two dicts, keyed by the same seeds.
  * Log-likelihoods use the same arithmetic as the WIS code (hurdle: gate + fired * magnitude, Gaussian in
    logit-dose space with the logit Jacobian), checked against the actor's own log_prob_of_action by
    check_loglik_matches.
  * Data loading lives in analysis/data.py; this file only computes on processors it is given.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit

from src.rl.analysis import data as D

DRUGS = {"fluid": ("f", 0), "vaso": ("v", 1)}          # name -> (actor output prefix, action column)
CLAMP = 1e-4                                            # dose clamp before the logit, as in the WIS code

# Seed colours: first five categorical slots of the project's validated palette (adjacent-pair CVD and
# normal-vision checks pass). Three of them are under 3:1 contrast on white, so every seed also gets its own
# marker and a legend entry; colour is never the only cue.
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEED_MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTE, GRID = "#222222", "#6b6b6b", "#e6e6e6"


# =============================================================================
# Per-row quantities
# =============================================================================

def _logit(x):
    x = np.clip(x, CLAMP, 1 - CLAMP)
    return np.log(x / (1 - x))


def bc_rows(model, p) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For one BC model on its processor: per drug, per RL row, the logged dose, whether it fired, BC's fire
    probability, BC's raw (logit-space) mean and std, the logged logit-dose and its z-score under BC.
    """
    out = model.actor_outputs(D.states(p))
    act = D.actions(p)
    thr = float(getattr(model.actor, "zero_threshold", D.NONZERO))
    rows = {}
    for drug, (k, col) in DRUGS.items():
        dose = act[:, col].astype(np.float64)
        mean = out[f"{k}_raw_mean"].astype(np.float64)
        std = out[f"{k}_std"].astype(np.float64)
        logit = _logit(dose)
        if f"{k}_gate_logit" in out:                     # exact: scored from the raw gate logit
            gl = out[f"{k}_gate_logit"].astype(np.float64)
            prob = expit(gl)
        else:                                            # older models.py: float32 probabilities only
            prob = out[f"{k}_p"].astype(np.float64)
            pc = np.clip(prob, 1e-12, 1 - 1e-12)
            gl = np.log(pc / (1 - pc))
        rows[drug] = {"dose": dose, "fired": dose > thr, "p": prob, "gate_logit": gl,
                      "raw_mean": mean, "std": std, "logit": logit, "z": (logit - mean) / std}
    return rows


def _gate_loglik(fired, gate_logit, smooth=0.0):
    """
    Gate log-likelihood from the raw logit. smooth=0 is the plain Bernoulli log-probability of the logged
    fire / no-fire. smooth>0 is what the BC actor's log_prob_of_action reports: the negative BCE against labels
    smoothed to y*(1-2s)+s (the actor uses s = 0.02), i.e. log p - s*logit when fired and log(1-p) + s*logit when not.
    """
    y = fired * (1 - 2 * smooth) + smooth
    return -(y * np.logaddexp(0, -gate_logit) + (1 - y) * np.logaddexp(0, gate_logit))


def _drug_loglik(fired, dose, logit, gate_logit, mean, std, smooth=0.0):
    """(gate log-lik, magnitude log-lik) per row; magnitude is 0 where the drug was not given. Scalars broadcast."""
    gate = _gate_loglik(fired, gate_logit, smooth)
    a = np.clip(dose, CLAMP, 1 - CLAMP)
    z = (logit - mean) / std
    mag = -0.5 * z ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi) - np.log(a * (1 - a) + 1e-6)
    return gate, np.where(fired, mag, 0.0)


def bc_loglik(rows, gate_smooth=0.0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    {drug: {'gate': ..., 'mag': ...}} per-row log-likelihood of the logged action under BC.
    gate_smooth=0: plain Bernoulli (a proper density; use for calibration and for comparing with the baseline).
    gate_smooth=actor.gate_smooth (0.02): exactly what actor.log_prob_of_action returns, i.e. the density WIS divides by.
    """
    out = {}
    for drug, r in rows.items():
        gate, mag = _drug_loglik(r["fired"], r["dose"], r["logit"], r["gate_logit"], r["raw_mean"], r["std"], gate_smooth)
        out[drug] = {"gate": gate, "mag": mag}
    return out


def marginal_reference(train_p, threshold=D.NONZERO) -> Dict[str, Dict[str, float]]:
    """
    The no-state baseline density, fitted on the TRAIN actions: per drug a constant fire rate and one Gaussian over
    the logit-dose of fired hours. BC has to beat this for its states to be telling WIS anything.
    """
    act = D.actions(train_p)
    ref = {}
    for drug, (_, col) in DRUGS.items():
        dose = act[:, col].astype(np.float64)
        fired = dose > threshold
        lg = _logit(dose[fired])
        ref[drug] = {"rate": float(fired.mean()), "mu": float(lg.mean()), "sd": float(lg.std())}
    return ref


def marginal_loglik(rows, ref, gate_smooth=0.0) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for drug, r in rows.items():
        m = ref[drug]
        rate = min(max(m["rate"], 1e-12), 1 - 1e-12)
        gate, mag = _drug_loglik(r["fired"], r["dose"], r["logit"], np.log(rate / (1 - rate)), m["mu"], m["sd"],
                                 gate_smooth)
        out[drug] = {"gate": gate, "mag": mag}
    return out


# =============================================================================
# Checks
# =============================================================================

def check_loglik_matches(models, procs, n=5000, tol=1e-3):
    """
    This file's per-row log-likelihood (summed over both drugs) against the actor's own log_prob_of_action on the
    first n rows of each seed. Raises on a mismatch, so the quantities below are known to be built from the density
    WIS divides by. The actor's gate term is label-smoothed (actor.gate_smooth); that smoothing is applied here for
    the comparison, and the average shift it causes relative to a plain Bernoulli gate is printed (per hour; a
    negative value raises every WIS log-ratio by that much, so it grows with stay length).
    """
    import torch
    for seed, m in models.items():
        p = procs[seed]
        rows = bc_rows(m, p)
        sm = float(getattr(m.actor, "gate_smooth", 0.0))
        ll = bc_loglik(rows, sm)
        plain = bc_loglik(rows, 0.0)
        shift = np.mean(sum(ll[d]["gate"] - plain[d]["gate"] for d in DRUGS))
        print(f"  seed {seed}: gate_smooth={sm}; mean shift of the behaviour log-lik from smoothing {shift:+.4f} nats/hour")
        mine = sum(ll[d]["gate"] + ll[d]["mag"] for d in DRUGS)[:n]
        s = torch.as_tensor(D.states(p)[:n], dtype=torch.float32, device=m.device)
        a = torch.as_tensor(D.actions(p)[:n], dtype=torch.float32, device=m.device)
        with torch.no_grad():
            theirs = m.actor.log_prob_of_action(s, a).reshape(-1).cpu().numpy().astype(np.float64)
        err = float(np.max(np.abs(mine - theirs)))
        print(f"  seed {seed}: max |difference| over {len(mine)} rows = {err:.2e}")
        if err > tol:
            raise ValueError(f"seed {seed}: log-likelihood here differs from actor.log_prob_of_action by {err:.3g}")


# =============================================================================
# Tables
# =============================================================================

def loglik_table(models, procs, ref, gate_smooth=0.0) -> pd.DataFrame:
    """
    Per seed and drug: BC against the no-state baseline, in nats. gate_* are per row; mag_* per fired row.
    gain_per_row = (gate gain) + fired share * (magnitude gain): what the states add per hour. Positive = BC beats
    the baseline. gate_smooth=0 (default) scores both as plain Bernoulli densities; pass 'actor' to score BC with
    its own label-smoothed gate (what WIS uses; the baseline is smoothed the same way).
    """
    recs = []
    for seed, m in models.items():
        rows = bc_rows(m, procs[seed])
        sm = float(m.actor.gate_smooth) if gate_smooth == "actor" else float(gate_smooth)
        bc, mg = bc_loglik(rows, sm), marginal_loglik(rows, ref, sm)
        for drug in DRUGS:
            f = rows[drug]["fired"]
            gate_gain = bc[drug]["gate"].mean() - mg[drug]["gate"].mean()
            mag_gain = (bc[drug]["mag"][f] - mg[drug]["mag"][f]).mean() if f.any() else np.nan
            recs.append({"seed": seed, "drug": drug, "fired_share": f.mean(),
                         "gate_ll_bc": bc[drug]["gate"].mean(), "gate_ll_base": mg[drug]["gate"].mean(),
                         "mag_ll_bc": bc[drug]["mag"][f].mean() if f.any() else np.nan,
                         "mag_ll_base": mg[drug]["mag"][f].mean() if f.any() else np.nan,
                         "gain_per_row": gate_gain + f.mean() * (0.0 if np.isnan(mag_gain) else mag_gain)})
    return pd.DataFrame(recs).set_index(["seed", "drug"])


def _auc(p, y):
    """Rank-based AUROC of p for the binary y."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(p)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gate_reliability(rows, drug, n_bins=10) -> pd.DataFrame:
    """Equal-count bins of predicted fire probability: mean prediction against the observed fire rate."""
    p, y = rows[drug]["p"], rows[drug]["fired"].astype(float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    recs = [{"bin": i, "n": int((b == i).sum()), "mean_pred": p[b == i].mean(), "obs_rate": y[b == i].mean()}
            for i in range(len(edges) - 1) if (b == i).any()]
    return pd.DataFrame(recs)


def gate_summary(models, procs, n_bins=10) -> pd.DataFrame:
    """
    Per seed and drug: what BC predicts at hours where the clinician gave / did not give the drug, plus Brier score,
    expected calibration error (equal-count bins) and AUROC.
    """
    recs = []
    for seed, m in models.items():
        rows = bc_rows(m, procs[seed])
        for drug in DRUGS:
            p, f = rows[drug]["p"], rows[drug]["fired"]
            rel = gate_reliability(rows, drug, n_bins)
            ece = float((rel["n"] * (rel["mean_pred"] - rel["obs_rate"]).abs()).sum() / rel["n"].sum())
            recs.append({"seed": seed, "drug": drug, "obs_rate": f.mean(), "mean_pred": p.mean(),
                         "p_at_zero_mean": p[~f].mean(), "p_at_zero_over_half": (p[~f] > 0.5).mean(),
                         "p_at_given_mean": p[f].mean(), "brier": float(np.mean((p - f) ** 2)),
                         "ece": ece, "auroc": _auc(p, f)})
    return pd.DataFrame(recs).set_index(["seed", "drug"])


def magnitude_summary(models, procs) -> pd.DataFrame:
    """
    Fired hours only, per seed and drug: the z-score of the logged logit-dose under BC's own mean and std. A
    calibrated density has mean ~0, sd ~1 and few |z| > 3. Also BC's std (floor / spread) and the share of doses
    sitting on the upper clamp (a pile-up there inflates z whatever BC does).
    """
    recs = []
    for seed, m in models.items():
        rows = bc_rows(m, procs[seed])
        for drug in DRUGS:
            r, f = rows[drug], rows[drug]["fired"]
            z, sd = r["z"][f], r["std"][f]
            recs.append({"seed": seed, "drug": drug, "n_fired": int(f.sum()), "z_mean": z.mean(), "z_sd": z.std(),
                         "frac_abs_z_gt3": (np.abs(z) > 3).mean(), "frac_abs_z_gt10": (np.abs(z) > 10).mean(),
                         "std_min": sd.min(), "std_p50": np.median(sd), "std_p95": np.percentile(sd, 95),
                         "frac_at_upper_clamp": (r["dose"][f] >= 1 - CLAMP).mean()})
    return pd.DataFrame(recs).set_index(["seed", "drug"])


def calibration_by_severity(models, procs, edges) -> pd.DataFrame:
    """
    Per seed, drug and severity bin (0 = sickest; edges from the TRAIN health score, see data.severity_edges):
    observed fire rate vs BC's mean fire probability, and for fired hours the observed mean logit-dose vs BC's mean,
    with the z-score mean / sd. Shows where in the severity range BC is off.
    """
    recs = []
    for seed, m in models.items():
        p = procs[seed]
        rows = bc_rows(m, p)
        sev = D.severity_bin(D.health_score(p)[0], edges)
        for drug in DRUGS:
            r = rows[drug]
            for b in range(len(edges) - 1):
                sel = sev == b
                if not sel.any():
                    continue
                f = sel & r["fired"]
                recs.append({"seed": seed, "drug": drug, "severity_bin": b, "n": int(sel.sum()),
                             "obs_rate": r["fired"][sel].mean(), "mean_pred": r["p"][sel].mean(),
                             "obs_logit": r["logit"][f].mean() if f.any() else np.nan,
                             "bc_logit": r["raw_mean"][f].mean() if f.any() else np.nan,
                             "z_mean": r["z"][f].mean() if f.any() else np.nan,
                             "z_sd": r["z"][f].std() if f.sum() > 1 else np.nan})
    return pd.DataFrame(recs)


def cross_seed_agreement(models, procs) -> Dict[str, pd.DataFrame]:
    """
    Spearman correlation, between seeds, of BC's fire probability on the same patient-hours. The rows must be the
    same hours (check_rows_aligned is applied; each seed still uses its own states, so this is what a change of
    state embedding plus BC seed does to the density).
    """
    D.check_rows_aligned(procs)
    probs = {s: bc_rows(m, procs[s]) for s, m in models.items()}
    seeds = list(models)
    out = {}
    for drug in DRUGS:
        mat = np.eye(len(seeds))
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                rho = stats.spearmanr(probs[seeds[i]][drug]["p"], probs[seeds[j]][drug]["p"])[0]
                mat[i, j] = mat[j, i] = rho
        out[drug] = pd.DataFrame(mat, index=seeds, columns=seeds)
    return out


def fluid_gate_wis_impact(models, procs, gate_smooth="actor") -> pd.DataFrame:
    """
    How much the fluid gate at zero-fluid hours could distort a stay's weight, per seed:
      p_no_fluid_*  : probability BC gives the logged 'no fluid' at zero-fluid hours (fluid), and the same for
                      'no vaso' at zero-vaso hours (reference: the better-calibrated gate)
      share_*       : per stay, the share of the total gate negative log-likelihood (both drugs) that comes from
                      zero-fluid hours (median / mean / p90 over stays), with the gate as WIS sees it
                      (gate_smooth='actor'); pass 0 for the plain Bernoulli gate
    """
    recs = []
    for seed, m in models.items():
        p = procs[seed]
        rows = bc_rows(m, p)
        f, v = rows["fluid"], rows["vaso"]
        sm = float(m.actor.gate_smooth) if gate_smooth == "actor" else float(gate_smooth)
        ll = bc_loglik(rows, sm)
        lp_f, lp_v = ll["fluid"]["gate"], ll["vaso"]["gate"]
        tid = D.trajectory_ids(p)
        total = np.bincount(tid, weights=-(lp_f + lp_v))
        zero_f = np.bincount(tid, weights=-lp_f * (~f["fired"]))
        share = (zero_f / total)[total > 1e-8]
        recs.append({"seed": seed,
                     "p_no_fluid_median": np.median(1 - f["p"][~f["fired"]]), "p_no_fluid_mean": (1 - f["p"][~f["fired"]]).mean(),
                     "p_no_vaso_median": np.median(1 - v["p"][~v["fired"]]), "p_no_vaso_mean": (1 - v["p"][~v["fired"]]).mean(),
                     "share_zero_fluid_median": np.median(share), "share_zero_fluid_mean": share.mean(),
                     "share_zero_fluid_p90": np.percentile(share, 90), "n_stays": len(share)})
    return pd.DataFrame(recs).set_index("seed")


# =============================================================================
# Plots
# =============================================================================

def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbdbd")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)


def _seed_style(i):
    return SEED_COLORS[i % len(SEED_COLORS)], SEED_MARKERS[i % len(SEED_MARKERS)]


def plot_gate_reliability(models, procs, n_bins=10, save_path=None):
    """Reliability of the fire probability, one panel per drug, one line per seed; histogram of predictions below."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    for i, (seed, m) in enumerate(models.items()):
        col, mk = _seed_style(i)
        rows = bc_rows(m, procs[seed])
        for j, drug in enumerate(DRUGS):
            rel = gate_reliability(rows, drug, n_bins)
            axes[0, j].plot(rel["mean_pred"], rel["obs_rate"], marker=mk, color=col, lw=1.4, ms=5, label=f"seed {seed}")
            axes[1, j].hist(rows[drug]["p"], bins=np.linspace(0, 1, 41), histtype="step", color=col, lw=1.2)
    for j, drug in enumerate(DRUGS):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=MUTE, lw=1, ls="--")
        ax.set_title(f"{drug} gate", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("mean predicted fire probability (equal-count bins)", color=INK)
        ax.set_ylabel("observed fire rate", color=INK)
        _style(ax)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("predicted fire probability", color=INK)
        axes[1, j].set_ylabel("hours", color=INK)
        _style(axes[1, j])
    axes[0, 0].legend(frameon=False, fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_magnitude_z(models, procs, lim=6, save_path=None):
    """Distribution of the logged logit-dose in BC's own units (fired hours). N(0,1) dashed; values beyond +-lim pile at the edge."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, constrained_layout=True)
    grid = np.linspace(-lim, lim, 400)
    for i, (seed, m) in enumerate(models.items()):
        col, _ = _seed_style(i)
        rows = bc_rows(m, procs[seed])
        for j, drug in enumerate(DRUGS):
            z = rows[drug]["z"][rows[drug]["fired"]]
            axes[j].hist(np.clip(z, -lim, lim), bins=np.linspace(-lim, lim, 61), density=True, histtype="step",
                         color=col, lw=1.3, label=f"seed {seed}")
    for j, drug in enumerate(DRUGS):
        axes[j].plot(grid, stats.norm.pdf(grid), color=MUTE, lw=1, ls="--", label="N(0,1)")
        axes[j].set_title(f"{drug} magnitude (fired hours)", loc="left", color=INK, fontsize=11)
        axes[j].set_xlabel(f"(logged logit-dose - BC mean) / BC std, clipped at +-{lim}", color=INK)
        _style(axes[j])
    axes[0].set_ylabel("density", color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_calibration_by_severity(tbl, save_path=None):
    """
    Rows = drug, columns = (fire rate, mean logit-dose of fired hours). Observed in dark ink, BC per seed in colour.
    x = severity bin, 0 = sickest.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True, constrained_layout=True)
    seeds = sorted(tbl["seed"].unique(), key=str)
    for j, drug in enumerate(DRUGS):
        d = tbl[tbl["drug"] == drug]
        obs = d.groupby("severity_bin")[["obs_rate", "obs_logit"]].mean()
        axes[j, 0].plot(obs.index, obs["obs_rate"], color=INK, lw=2.2, label="observed")
        axes[j, 1].plot(obs.index, obs["obs_logit"], color=INK, lw=2.2, label="observed")
        for i, s in enumerate(seeds):
            col, mk = _seed_style(i)
            ds = d[d["seed"] == s]
            axes[j, 0].plot(ds["severity_bin"], ds["mean_pred"], color=col, marker=mk, ms=4, lw=1.2, label=f"BC seed {s}")
            axes[j, 1].plot(ds["severity_bin"], ds["bc_logit"], color=col, marker=mk, ms=4, lw=1.2)
        axes[j, 0].set_ylabel(f"{drug}: fire rate", color=INK)
        axes[j, 1].set_ylabel(f"{drug}: mean logit-dose (fired hours)", color=INK)
        for k in (0, 1):
            _style(axes[j, k])
    for k in (0, 1):
        axes[1, k].set_xlabel("severity bin (0 = sickest)", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig