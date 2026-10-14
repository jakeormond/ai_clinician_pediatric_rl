"""
src/rl/evaluate_cql_alpha_sweep.py

CQL alpha sweep analysis, run on the VAL split only (alpha selection is a modeling decision;
the test set stays reserved for the final 5-seed evaluation).

Loading is NOT done here. Runs, models and processors come from src.rl.analysis:
    runs.py    make_study(...)           -> Study {alpha: {seed: RunRecord}}
    models.py  load_study_models(study)  -> {(alpha,): {seed: LoadedModel}}
    data.py    DataStore                 -> processors built from each run's own data identity
This file holds only what is specific to the alpha sweep.

For every (alpha, seed) it computes:
  1. Own-critic advantage: Q_own(actor_action) - Q_own(clinician_action)
     -- an exploitation / red-flag check, not positive evidence on its own.
  2. SARSA-critic advantage: Q_sarsa(actor_action) - Q_sarsa(clinician_action)
     -- the primary evidence: an independent, unregularised critic corroborating (or not)
     that the actor found a real improvement.
  3. Occupancy support of the actor's ACTUAL chosen actions: the fraction of states whose
     (severity, fluid, vaso) combination was well supported (count >= min_count) in that
     seed's training data.

Seed matching. Each seed has its own Dynamic-DeepHit state embedding, so a critic can only
judge actions on states from its own embedding. CQL seed k is therefore judged by the SARSA
critic of seed k, on seed k's val states (checked against the run records, raises on mismatch).
The spread across seeds at each alpha is the spread across these 5 matched (CQL, SARSA) pairs.

Results are a tidy DataFrame, one row per (alpha, seed); summarize_by_alpha() aggregates it.
sarsa_stability() / stability_sensitivity() / plot_sarsa_stability() implement the SARSA-advantage
stability gate used in alpha selection; select_final_alpha() runs the full selection cascade.

Typical use (notebook):
    store = D.DataStore(keep=("train", "val"))
    df, fig = eval_cql.run_alpha_sweep_analysis(cql_models, sarsa_models, store)
    eval_cql.summarize_by_alpha(df)
    eval_cql.plot_combined_alpha_sweep(df, normalized=True, log_scale=True)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kneed import KneeLocator

from src.rl.analysis import data as D
from src.rl.evaluate_terminal_reward_models import build_joint_state_action_occupancy_counts, _bin_index


OWN_COLOR = "#c0392b"
SARSA_COLOR = "#2471a3"
OCC_COLOR = "#229954"
KNEE_COLOR = "#7d3c98"

# columns of the per-(alpha, seed) results table that summarize_by_alpha aggregates
METRICS = ("own_adv", "own_adv_norm", "sarsa_adv", "sarsa_adv_norm", "occupancy_frac")


# =============================================================================
# Pairing CQL runs with their SARSA judges
# =============================================================================

def _single_group(models: Dict) -> Dict:
    """{(): {seed: model}} (a Study with no group_by) -> {seed: model}; a plain {seed: model} passes through."""
    return models[()] if () in models else models


def _models_by_alpha(cql_models: Dict) -> Dict[float, Dict[int, Any]]:
    """{alpha: {seed: LoadedModel}}, with alpha read from each run's own cql_alpha."""
    out: Dict[float, Dict[int, Any]] = {}
    for key, seeds in cql_models.items():
        alphas = {m.record.get("cql_alpha") for m in seeds.values()}
        if len(alphas) != 1 or None in alphas:
            raise ValueError(f"group {key}: expected one cql_alpha across seeds, found {alphas}. "
                             f"Build the study with group_by=('cql_alpha',).")
        alpha = alphas.pop()
        if alpha in out:
            raise ValueError(f"cql_alpha={alpha} appears in more than one group")
        out[alpha] = dict(seeds)
    return dict(sorted(out.items()))


def check_judge_matches(model, judge):
    """Raise unless `judge` (a SARSA LoadedModel) has the same seed and state embedding as `model`."""
    if model.seed != judge.seed:
        raise ValueError(f"{model.record.label} paired with {judge.record.label}: different seeds")
    a, b = model.record.data_identity, judge.record.data_identity
    norm = lambda p: None if p is None else os.path.normpath(str(p))
    if (norm(a.state_source), a.state_seed) != (norm(b.state_source), b.state_seed):
        raise ValueError(f"{model.record.label} and {judge.record.label} were trained on different state "
                         f"embeddings:\n  {a.short()}\n  {b.short()}")


# =============================================================================
# Actor actions
# =============================================================================

@torch.no_grad()
def sample_actor_actions(model, states, batch_size=4096, sample_seed=0) -> np.ndarray:
    """
    Stochastic actions, sampled the way the actor was TRAINED (deterministic=False,
    hard_gate=True), rather than the deterministic mean / hard-threshold action. Use this to
    check whether a negative own-critic advantage under deterministic evaluation is an artifact
    of evaluating a stochastically-trained policy at a point training never optimised.
    Re-seeds torch with sample_seed so results are reproducible.
    """
    model.actor.eval()
    torch.manual_seed(sample_seed)
    states = np.asarray(states)
    out = []
    for i in range(0, len(states), batch_size):
        s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=model.device)
        actions, _, _ = model.actor(s, deterministic=False, hard_gate=True)
        out.append(actions.detach().cpu())
    return torch.cat(out).numpy()


def actor_actions(model, states, stochastic=False, sample_seed=0, batch_size=4096) -> np.ndarray:
    """(N, 2) actions of a CQL actor: deterministic (default) or stochastic samples."""
    if stochastic:
        return sample_actor_actions(model, states, batch_size=batch_size, sample_seed=sample_seed)
    return model.actor_outputs(states, batch_size=batch_size)["actions"]


# =============================================================================
# Advantage
# =============================================================================

def _iqr(q: np.ndarray) -> float:
    return max(float(np.quantile(q, 0.75) - np.quantile(q, 0.25)), 1e-6)


def compute_model_advantage(model, judge, processor, stochastic=False, sample_seed=0, batch_size=4096) -> Dict[str, Any]:
    """
    Per-timestep advantage of one CQL model's actions over the clinician's, on `processor`,
    judged by the model's own critic and by `judge` (the seed-matched SARSA LoadedModel).

    Returns RAW and IQR-NORMALISED advantage for each critic:
      own_advantage / own_advantage_normalized
      sarsa_advantage / sarsa_advantage_normalized
      own_q_iqr / sarsa_q_iqr, actor_actions
    Each advantage is divided by the IQR of THAT critic's Q-values under the real clinician
    actions (scale-free, so comparable across critics and across reward types on different
    absolute Q scales).
    """
    s = D.states(processor)
    a_clin = D.actions(processor)
    a_actor = actor_actions(model, s, stochastic=stochastic, sample_seed=sample_seed, batch_size=batch_size)

    out: Dict[str, Any] = {"actor_actions": a_actor}
    for name, critic_model in (("own", model), ("sarsa", judge)):
        q_clin = critic_model.q(s, a_clin, batch_size=batch_size)
        q_actor = critic_model.q(s, a_actor, batch_size=batch_size)
        adv = q_actor - q_clin
        iqr = _iqr(q_clin)
        out[f"{name}_advantage"] = adv
        out[f"{name}_advantage_normalized"] = adv / iqr
        out[f"{name}_q_iqr"] = iqr
    return out


# =============================================================================
# Occupancy support
# =============================================================================

@dataclass
class OccupancyTable:
    """Joint (severity x fluid-bin x vaso-bin) training-data counts, with the bin edges."""
    counts: Any
    severity_edges: Any
    fluid_edges: Any
    vaso_edges: Any


def build_occupancy_table(train_processor, n_severity_bins=20, n_action_bins=10) -> OccupancyTable:
    """Occupancy counts from one TRAIN processor (health score and actions aligned to RL rows)."""
    train_hs, _ = D.health_score(train_processor)
    train_actions = torch.as_tensor(D.actions(train_processor))
    counts, sev_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
        train_hs, train_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins)
    return OccupancyTable(counts, sev_edges, fluid_edges, vaso_edges)


def occupancy_support_of_actions(actions, health_score, table: OccupancyTable, min_count=10):
    """
    Fraction of timesteps where the actor's ACTUAL chosen (fluid, vaso) combination, given that
    timestep's own severity bin, was well supported (count >= min_count) in the training data.
    Returns (fraction, per-timestep counts).
    """
    actions = np.asarray(actions)
    hs = np.asarray(health_score)
    sev_bin = _bin_index(hs, table.severity_edges)
    fluid_bin = _bin_index(actions[:, 0], table.fluid_edges)
    vaso_bin = _bin_index(actions[:, 1], table.vaso_edges)
    counts = table.counts[sev_bin, fluid_bin, vaso_bin]
    return (counts >= min_count).mean(), counts


# =============================================================================
# The sweep
# =============================================================================

def run_alpha_sweep_analysis(cql_models, sarsa_models, store, split="val",
                             n_severity_bins=20, n_action_bins=10, min_count=10,
                             stochastic=False, sample_seed=0, verbose=True) -> pd.DataFrame:
    """
    cql_models   : load_study_models(study) for a study grouped by cql_alpha, {(alpha,): {seed: LoadedModel}}
    sarsa_models : load_study_models(sarsa_study), {(): {seed: LoadedModel}}; one critic per seed
    store        : data.DataStore. Use DataStore(keep=("train", "val")) so each seed's data loader
                   runs once (the train split is needed for the occupancy table).
    split        : split to evaluate on. Keep "val" for alpha selection.
    stochastic   : evaluate advantage on stochastic action samples (matching training) rather
                   than the deterministic action.

    Returns a DataFrame, one row per (alpha, seed):
      alpha, seed, own_adv, own_adv_norm, sarsa_adv, sarsa_adv_norm, occupancy_frac,
      own_q_iqr, sarsa_q_iqr, n_rows
    where *_adv are mean per-timestep advantages (raw Q units) and *_adv_norm the same divided
    by the matching critic's Q IQR under clinician actions.
    """
    sarsa = _single_group(sarsa_models)
    by_alpha = _models_by_alpha(cql_models)

    needed = {s for seeds in by_alpha.values() for s in seeds}
    missing = sorted(needed - set(sarsa), key=str)
    if missing:
        raise KeyError(f"no SARSA critic for seeds {missing}; SARSA has seeds {sorted(sarsa, key=str)}")

    # Occupancy tables and severity depend on the seed's data, not on alpha: build once per data identity.
    tables: Dict[Any, OccupancyTable] = {}
    val_hs: Dict[Any, np.ndarray] = {}

    def data_key(m):
        return (m.record.data_identity.key, D.reward_signature(m.record))

    rows = []
    for alpha, seeds in by_alpha.items():
        t0 = time.time()
        for seed in sorted(seeds):
            m, judge = seeds[seed], sarsa[seed]
            check_judge_matches(m, judge)
            p = store.processor(m, split)
            k = data_key(m)
            if k not in tables:
                tables[k] = build_occupancy_table(store.processor(m, "train"), n_severity_bins, n_action_bins)
                val_hs[k] = D.health_score(p)[0]

            adv = compute_model_advantage(m, judge, p, stochastic=stochastic, sample_seed=sample_seed)
            occ, _ = occupancy_support_of_actions(adv["actor_actions"], val_hs[k], tables[k], min_count)
            rows.append({
                "alpha": alpha, "seed": seed,
                "own_adv": adv["own_advantage"].mean(),
                "own_adv_norm": adv["own_advantage_normalized"].mean(),
                "sarsa_adv": adv["sarsa_advantage"].mean(),
                "sarsa_adv_norm": adv["sarsa_advantage_normalized"].mean(),
                "occupancy_frac": float(occ),
                "own_q_iqr": adv["own_q_iqr"], "sarsa_q_iqr": adv["sarsa_q_iqr"],
                "n_rows": D.n_rows(p),
            })
        if verbose:
            r = pd.DataFrame([x for x in rows if x["alpha"] == alpha])
            print(f"alpha={alpha:<10g} ({'stochastic' if stochastic else 'deterministic'}, {split}, {len(r)} seeds)  "
                  f"own_norm={r.own_adv_norm.mean():+.4f}  sarsa_norm={r.sarsa_adv_norm.mean():+.4f} "
                  f"[{r.sarsa_adv_norm.min():+.4f}, {r.sarsa_adv_norm.max():+.4f}]  "
                  f"occupancy={r.occupancy_frac.mean() * 100:.1f}%  ({time.time() - t0:.1f}s)", flush=True)

    return pd.DataFrame(rows).sort_values(["alpha", "seed"]).reset_index(drop=True)


def summarize_by_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """Per-alpha mean / min / max / std across seeds for every metric (columns: (metric, stat)), plus n_seeds."""
    g = df.groupby("alpha")
    summ = g[list(METRICS)].agg(["mean", "min", "max", "std"])
    summ["n_seeds"] = g["seed"].nunique()
    return summ


# =============================================================================
# Plots
# =============================================================================

def _line_with_band(ax, summ, metric, scale, **kw):
    x = summ.index.values
    mean, lo, hi = (summ[(metric, s)].values * scale for s in ("mean", "min", "max"))
    line, = ax.plot(x, mean, marker=kw.pop("marker", "o"), **kw)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15, linewidth=0)
    return line


def _alpha_filter(df, alpha_range=None, log_scale=False):
    if alpha_range is not None:
        lo, hi = alpha_range
        df = df[(df.alpha >= lo) & (df.alpha <= hi)]
    if log_scale and (df.alpha == 0).any():
        print("Note: alpha=0 excluded from plot (log scale can't show it).")
        df = df[df.alpha != 0]
    if df.empty:
        raise ValueError("No alpha values remain after applying alpha_range/log_scale filtering.")
    return df


def plot_alpha_sweep(df, normalized=True, log_scale=True, alpha_range=None, save_path=None):
    """
    Three panels vs alpha: own-critic advantage, SARSA-critic advantage, occupancy support.
    Line = mean across seeds, band = min-max across seeds. normalized=True plots advantage as a
    fraction of the critic's own Q IQR (comparable across alphas and reward types).
    """
    df = _alpha_filter(df, alpha_range, log_scale)
    summ = summarize_by_alpha(df)
    sfx = "_norm" if normalized else ""
    unit = "(fraction of critic's own Q IQR)" if normalized else "(raw Q units)"
    n_seeds = int(summ["n_seeds"].min())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _line_with_band(axes[0], summ, f"own_adv{sfx}", 1, color=OWN_COLOR)
    axes[0].set_ylabel(f"Own-critic advantage\n{unit}")
    axes[0].set_title("Own-critic advantage\n(exploitation red-flag, not positive evidence)", fontsize=10)

    _line_with_band(axes[1], summ, f"sarsa_adv{sfx}", 1, color=SARSA_COLOR)
    axes[1].set_ylabel(f"SARSA-critic advantage\n{unit}")
    axes[1].set_title("SARSA-critic advantage\n(primary evidence of real improvement)", fontsize=10)

    _line_with_band(axes[2], summ, "occupancy_frac", 100, color=OCC_COLOR)
    axes[2].set_ylabel("% of actions well-supported")
    axes[2].set_title("Occupancy support of\nactor's chosen actions", fontsize=10)

    for i, ax in enumerate(axes):
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel("cql_alpha")
        if i < 2:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].text(0.02, 0.02, f"mean and min-max across {n_seeds} seeds", transform=axes[1].transAxes,
                 fontsize=8, color="grey")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def find_knee_alpha(df, metric="sarsa_advantage", curve="convex", direction="decreasing", S=1.0,
                    alpha_min=None, normalized=False):
    """
    Kneedle knee/elbow of a metric vs alpha, on the across-seed MEAN.

    metric: "sarsa_advantage" (default) | "own_advantage" | "occupancy" (this one increases, so
        pass direction="increasing"). normalized=True uses the IQR-normalised advantage (no
        effect on occupancy, which is already a %).
    curve/direction: passed to KneeLocator; must match the actual shape of the metric.
    alpha_min: restrict the fit to alphas >= alpha_min (e.g. to skip an early noisy dip).
    S: KneeLocator sensitivity; higher finds fewer, more conservative knees.
    Returns the alpha at the knee, or None if Kneedle finds none.
    """
    cols = {"sarsa_advantage": "sarsa_adv", "own_advantage": "own_adv", "occupancy": "occupancy_frac"}
    if metric not in cols:
        raise ValueError(f"Unknown metric '{metric}'")
    col = cols[metric] + ("_norm" if normalized and metric != "occupancy" else "")
    mean = df.groupby("alpha")[col].mean()
    if alpha_min is not None:
        mean = mean[mean.index >= alpha_min]
    kl = KneeLocator(list(mean.index), list(mean.values), curve=curve, direction=direction, S=S)
    return kl.elbow


def plot_combined_alpha_sweep(df, title="", save_path=None, mark_knee=True, normalized=False,
                              log_scale=False, alpha_range=None):
    """
    Both advantage curves (left axis) and occupancy support (right axis) on one plot.
    Lines = mean across seeds, bands = min-max across seeds.

    normalized : plot IQR-normalised advantage (comparable across critics/reward types).
    log_scale  : log x-axis; alpha=0 cannot be shown and is dropped from the plot.
    alpha_range: optional (min, max) to zoom into a sub-range of a wide sweep.
    mark_knee  : Kneedle knee of the SARSA advantage, computed over the plotted alpha range only.
    """
    df = _alpha_filter(df, alpha_range, log_scale)
    summ = summarize_by_alpha(df)
    sfx = "_norm" if normalized else ""
    n_seeds = int(summ["n_seeds"].min())

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    l1 = _line_with_band(ax1, summ, f"own_adv{sfx}", 1, color=OWN_COLOR, label="Own-critic advantage")
    l2 = _line_with_band(ax1, summ, f"sarsa_adv{sfx}", 1, color=SARSA_COLOR, label="SARSA-critic advantage")
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_xlabel(r"cql_$\alpha$" + (" (log scale)" if log_scale else ""))
    ax1.set_ylabel("Advantage (fraction of critic's own Q IQR)" if normalized else "Advantage (raw Q units)")
    ax1.spines[["top"]].set_visible(False)

    ax2 = ax1.twinx()
    l3 = _line_with_band(ax2, summ, "occupancy_frac", 100, color=OCC_COLOR, marker="s", linestyle="--",
                         label="Occupancy support (%)")
    ax2.set_ylabel("% of chosen actions well-supported", color=OCC_COLOR)
    ax2.tick_params(axis="y", colors=OCC_COLOR)
    ax2.spines["right"].set_color(OCC_COLOR)
    ax2.spines[["top"]].set_visible(False)
    if log_scale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")

    lines = [l1, l2, l3]
    if mark_knee:
        knee_alpha = find_knee_alpha(df, normalized=normalized)
        if knee_alpha is not None:
            lines.append(ax1.axvline(knee_alpha, color=KNEE_COLOR, linewidth=1.5, linestyle=":",
                                     label=f"Knee (Kneedle): $\\alpha$={knee_alpha}"))
        else:
            print("Kneedle did not identify a knee for this curve.")

    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=False, fontsize=9)
    default_title = "CQL alpha sweep: advantage vs. occupancy support trade-off"
    ax1.set_title(title or (default_title + (" (normalized)" if normalized else "")), fontsize=12, fontweight="bold")
    ax1.text(0.02, 0.02, f"mean and min-max across {n_seeds} seeds", transform=ax1.transAxes, fontsize=8, color="grey")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


# =============================================================================
# SARSA-advantage stability (selection step 2: reliability gate)
# =============================================================================

def _window_ratio(values) -> float:
    """max/min of a window of advantages; inf if the window is not strictly positive (advantage not reliably > 0)."""
    lo, hi = float(np.min(values)), float(np.max(values))
    return np.inf if lo <= 0 else hi / lo


def sarsa_stability(df, window=3, max_ratio=2.0, normalized=True):
    """
    Sliding-window ratio test on the across-seed MEAN SARSA advantage vs alpha.

    Alphas are sorted; for every `window` consecutive alphas the ratio max/min of the mean
    advantage is computed (inf if any value in the window is <= 0). The first window whose ratio
    exceeds `max_ratio` marks the onset of instability, taken as that window's LOWEST alpha (the
    conservative choice); every alpha >= onset is flagged unstable.

    Returns (table, onset_alpha). The table has one row per alpha:
      mean, std (across seeds), n_seeds, n_seeds_positive, cv (std/|mean|),
      worst_window_ratio (largest ratio among windows containing that alpha), unstable.
    onset_alpha is None if no window fails.
    """
    col = "sarsa_adv_norm" if normalized else "sarsa_adv"
    g = df.groupby("alpha")[col]
    tab = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n_seeds": g.size(),
                        "n_seeds_positive": g.agg(lambda x: int((x > 0).sum()))})
    tab["cv"] = tab["std"] / tab["mean"].abs()
    v, n = tab["mean"].values, len(tab)
    if n < window:
        raise ValueError(f"only {n} alphas; need at least window={window}")
    ratios = np.array([_window_ratio(v[i:i + window]) for i in range(n - window + 1)])
    worst = np.ones(n)
    for i, r in enumerate(ratios):
        worst[i:i + window] = np.maximum(worst[i:i + window], r)
    tab["worst_window_ratio"] = worst
    bad = np.flatnonzero(ratios > max_ratio)
    onset = tab.index[bad[0]] if len(bad) else None
    tab["unstable"] = False if onset is None else (tab.index >= onset)
    return tab, onset


def stability_sensitivity(df, windows=(3, 4, 5), ratios=(1.25, 1.5, 2, 3, 5), normalized=True) -> pd.DataFrame:
    """Onset alpha (NaN = none) for a grid of window sizes and ratio thresholds: shows how much the gate depends on them."""
    out = pd.DataFrame(index=pd.Index(windows, name="window"), columns=pd.Index(ratios, name="max_ratio"), dtype=float)
    for w in windows:
        for r in ratios:
            onset = sarsa_stability(df, window=w, max_ratio=r, normalized=normalized)[1]
            out.loc[w, r] = np.nan if onset is None else onset
    return out


def plot_sarsa_stability(df, onset=None, normalized=True, log_scale=True, alpha_range=None, save_path=None):
    """SARSA advantage vs alpha: every seed (thin grey), mean and min-max band, and the instability onset."""
    df = _alpha_filter(df, alpha_range, log_scale)
    col = "sarsa_adv_norm" if normalized else "sarsa_adv"
    summ = summarize_by_alpha(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (seed, d) in enumerate(df.groupby("seed")):
        ax.plot(d["alpha"], d[col], color="grey", alpha=0.5, linewidth=0.8, label="individual seeds" if i == 0 else None)
    _line_with_band(ax, summ, col, 1, color=SARSA_COLOR, label="mean across seeds (band: min-max)")
    if onset is not None and onset >= df["alpha"].min():
        ax.axvline(onset, color=KNEE_COLOR, linewidth=1.5, linestyle=":", label=f"instability onset: alpha={onset:g}")
        ax.axvspan(onset, df["alpha"].max(), color=OWN_COLOR, alpha=0.08, linewidth=0)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(r"cql_$\alpha$" + (" (log scale)" if log_scale else ""))
    ax.set_ylabel("SARSA-critic advantage (fraction of critic's own Q IQR)" if normalized else "SARSA-critic advantage (raw Q units)")
    ax.set_title("SARSA-advantage stability across the alpha sweep", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


# =============================================================================
# Final alpha selection (the full cascade, on the results of run_alpha_sweep_analysis)
# =============================================================================

def select_final_alpha(df, stability_window=3, stability_max_ratio=2.0, own_rule="any_seed",
                       occupancy_margin=0.005, normalized=True, verbose=True):
    """
    Pre-specified selection cascade. Use a df computed on the VAL split.

      1. Own-critic validity gate: drop alphas whose own-critic advantage is <= 0
         (own_rule="any_seed": any seed <= 0 drops the alpha; "mean": the across-seed mean <= 0).
      2. SARSA stability gate: drop alphas at or above the onset found by sarsa_stability
         (window=stability_window, max_ratio=stability_max_ratio).
      3. Occupancy, safety-first but only "reasonable": among the survivors, every alpha whose mean
         occupancy support is within occupancy_margin (a fraction: 0.005 = 0.5 percentage points)
         of the best survivor counts as tied.
      4. Among the tied alphas, pick the highest mean SARSA advantage.

    Returns (final_alpha, info): info has one row per alpha with the quantities and pass/fail flag
    of every stage and which alpha was selected.
    """
    if own_rule not in ("any_seed", "mean"):
        raise ValueError("own_rule must be 'any_seed' or 'mean'")
    sarsa_col = "sarsa_adv_norm" if normalized else "sarsa_adv"
    own = df.groupby("alpha")["own_adv_norm"]
    info = pd.DataFrame({"own_adv_mean": own.mean(), "own_adv_min": own.min()})
    info["pass_own"] = (info["own_adv_min"] > 0) if own_rule == "any_seed" else (info["own_adv_mean"] > 0)

    stab, onset = sarsa_stability(df, window=stability_window, max_ratio=stability_max_ratio, normalized=normalized)
    info["pass_stability"] = ~stab["unstable"]
    info["sarsa_adv"] = df.groupby("alpha")[sarsa_col].mean()
    info["occupancy"] = df.groupby("alpha")["occupancy_frac"].mean()

    info["survivor"] = info["pass_own"] & info["pass_stability"]
    if not info["survivor"].any():
        raise ValueError("no alpha survives the own-critic and stability gates")
    best_occ = info.loc[info["survivor"], "occupancy"].max()
    info["tied_occupancy"] = info["survivor"] & (info["occupancy"] >= best_occ - occupancy_margin - 1e-12)
    final = info.loc[info["tied_occupancy"], "sarsa_adv"].idxmax()
    info["selected"] = info.index == final

    if verbose:
        print(f"alphas: {len(info)}")
        print(f"1. own-critic gate ({own_rule}): {int(info['pass_own'].sum())} pass; "
              f"dropped {list(info.index[~info['pass_own']])}")
        print(f"2. SARSA stability gate (window={stability_window}, max_ratio={stability_max_ratio}): "
              f"onset={onset}; {int((info['pass_stability'] & info['pass_own']).sum())} survive both gates")
        print(f"3. occupancy: best survivor {best_occ * 100:.2f}%, margin {occupancy_margin * 100:.2f} points -> "
              f"{int(info['tied_occupancy'].sum())} tied: {list(info.index[info['tied_occupancy']])}")
        print(f"4. highest SARSA advantage among the tied: alpha={final:g} "
              f"(SARSA adv {info.loc[final, 'sarsa_adv']:.4f}, occupancy {info.loc[final, 'occupancy'] * 100:.2f}%)")
    return final, info