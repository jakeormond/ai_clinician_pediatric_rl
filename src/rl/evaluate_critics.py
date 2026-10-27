"""
src/rl/evaluate_critics.py

What do the critics say about fluid dose, by patient severity?  (dose-response curves)

  res = dose_response({"SARSA health score": sarsa_models,
                       "SARSA terminal": sarsa_terminal_models,
                       "CQL health score": cql_models},
                      store, ref="SARSA health score", split="test", dim=0)
  summarize_shapes(res)          # shape fractions and direction statistic: mean / min / max across seeds
  plot_dose_response(res)        # curves: one row per critic, one column per severity bin
  plot_shape_fractions(res)      # share of rows whose curve is lower / interior / higher / irregular / flat
  obs = observed_by_dose(res, critics, store, names=[...], ref=...)   # model-free reference, see the last section
  plot_dose_response(res, observed=obs)                               # the same curves with the observed points on top

What is computed. For a critic and one patient-hour (state s, logged action a), replace the dose of one action
dimension (fluid by default) by d, keep the other dimension at its logged value, and read Q:

    delta(s, d) = [ Q(s, a with dim = d) - Q(s, a with dim = 0) ] / scale

scale is the IQR of that critic's Q at the clinician's actions over all rows of the split, so critics trained on
different rewards (health-score change vs terminal outcome) are compared by SHAPE, not by their own units.
delta(s, 0) = 0 by construction. The curve of a severity bin is the mean of delta over the rows in the bin (all rows
pooled, including the many where the clinician gave no fluid). Severity bins are equal-count bins of the health score
(dose_histogram_by_alpha.severity_bins), low health score = sicker.

Support. Q at a dose that was never given in that kind of state is extrapolation (for CQL it is also shaped by the
conservatism penalty). Each bin's curve is therefore only read up to the `support_q` quantile of the clinician's nonzero
doses in that bin. The grid beyond that stays in the table (supported = False) but is neither plotted nor classified.

Shape of one row's supported curve c (c[0] = 0, last point = highest supported dose), with tolerance tol = flat_tol:
    flat       max(c) - min(c) < tol
    irregular  two or more turning points, a turning point being a reversal of direction larger than tol
    interior   max(c) exceeds both end points by at least tol (an inverted U)
    higher     otherwise, and c[-1] >= c[0]  (more fluid is better)
    lower      otherwise                      (less fluid is better)
(checked in that order; a U-shaped curve, whose best dose is at an end, counts as higher or lower)
Direction statistic delta_top = delta at the highest supported dose, averaged over the bin's rows: > 0 means the critic
prefers that dose to no fluid, < 0 the reverse.

Pre-specified settings (state them in the methods; do not tune them after looking at results):
n_bins = 5, n_grid = 21, support_q = 0.95, flat_tol = 0.02. Seeds are kept separate throughout; plotted bands are
min-max across seeds.

Reading. This shows what each critic learned, not the causal effect of fluid: Q reflects how clinicians dosed, and a
SARSA critic is the value of giving this dose now and then following clinician behaviour. Every critic is assumed to
have higher Q = better outcome.

Observed reference (observed_by_dose). A SARSA critic estimates what actually followed each dose in the clinician data,
so the data themselves give a model-free check on its slope. Within each severity bin the rows are split into a no-fluid
group and `n_groups` groups of nonzero supported doses (equal-count tertiles of the nonzero doses, edges shared by all
bins). For each dose group the observed discounted return-to-go (the critic's own reward and discount) is compared with
that of the no-fluid group, in the same IQR units as the curves. To limit confounding by severity the comparison is made
inside `n_fine` finer health-score strata per bin and averaged with weights equal to the dose group's row counts. Only
interior rows are used (at least interior_horizon later rows in the stay, as in the SARSA level check), so the return
is nearly fully observed. The interval is a bootstrap over stays. It adjusts for the health score only: the critic
conditions on the whole state, and clinicians dose by indication, so the two are not expected to agree exactly, and
neither is the causal effect of fluid.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rl.analysis import data as D
from src.rl.dose_histogram_by_alpha import severity_bins
from src.rl.evaluate_cql_alpha_sweep import check_judge_matches
from src.rl.evaluate_fqe_final_seeds import _by_seed, discounted_return_to_go, interior_horizon, rows_after_in_stay

DIM_NAMES = ("fluid", "vaso")                    # columns of the action array
SHAPES = ("lower", "interior", "higher", "irregular", "flat")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]      # reference palette slots 1-3 (validated for any pair of three)
SHAPE_COLORS = {"lower": "#eb6834", "interior": "#1baf7a", "higher": "#2a78d6", "irregular": "#6250d6",
                "flat": "#c9c8c2"}
GREY = "#8a8985"


@dataclass
class DoseResponse:
    curves: pd.DataFrame        # critic, seed, bin, dose, delta (mean over the bin's rows), supported
    shapes: pd.DataFrame        # critic, seed, bin, n, lower, interior, higher, flat (fractions), delta_top
    labels: List[str]           # health-score range of each bin, ascending (first = sickest)
    support: np.ndarray         # highest supported dose of each bin
    grid: np.ndarray            # doses swept
    scales: pd.DataFrame        # critic, seed, iqr (the divisor used for that critic)
    dim: int
    split: str
    settings: dict


# =============================================================================
# Computation
# =============================================================================

def _turning_points(c: np.ndarray, tol: float) -> np.ndarray:
    """
    Number of turning points of each row of c (n, k): reversals of direction larger than tol (a zigzag filter, so
    changes smaller than tol never count). Vectorised over rows, looping over the k grid points.
    """
    n, k = c.shape
    ext = c[:, 0].copy()                 # latest extreme in the current direction (the start value until a direction is set)
    direction = np.zeros(n, dtype=int)   # 0 not yet set, +1 rising, -1 falling
    turns = np.zeros(n, dtype=int)
    for j in range(1, k):
        v = c[:, j]
        rising, falling, unset = direction == 1, direction == -1, direction == 0
        new_up = unset & (v - ext >= tol)
        new_dn = unset & (ext - v >= tol)
        flip_dn = rising & (ext - v >= tol)
        flip_up = falling & (v - ext >= tol)
        extend = (rising & (v > ext)) | (falling & (v < ext))
        turns += (flip_dn | flip_up)
        direction = np.where(new_up | flip_up, 1, np.where(new_dn | flip_dn, -1, direction))
        ext = np.where(new_up | new_dn | flip_dn | flip_up | extend, v, ext)
    return turns


def _classify(c: np.ndarray, tol: float) -> np.ndarray:
    """
    Shape index of each row of c (n, k), k >= 2, c[:, 0] = 0, in the order of SHAPES:
    0 lower, 1 interior, 2 higher, 3 irregular, 4 flat. Precedence: flat, irregular, interior, then higher / lower.
    """
    ends = np.maximum(c[:, 0], c[:, -1])
    out = np.where(c[:, -1] >= c[:, 0], 2, 0)
    out = np.where(c.max(1) - ends >= tol, 1, out)
    out = np.where(_turning_points(c, tol) >= 2, 3, out)
    return np.where(c.max(1) - c.min(1) < tol, 4, out)


def dose_response(critic_sets: Dict[str, Dict], store, ref: str, split: str = "test", dim: int = 0, n_bins: int = 5,
                  n_grid: int = 21, support_q: float = 0.95, flat_tol: float = 0.02, reduction: str = "mean",
                  batch_size: int = 4096, verbose: bool = True) -> DoseResponse:
    """
    critic_sets : {name: {seed: LoadedModel}} (or a Study result with one group), e.g. SARSA health-score critics,
                  SARSA terminal-reward critics, CQL. Seeds present in every set are used.
    ref         : the name in critic_sets whose data defines the rows (states, clinician actions, health score).
                  Every other set must share its seed and state embedding (checked); it is evaluated on those states.
    dim         : action dimension swept (0 = fluid, 1 = vasopressor).
    """
    sets = {k: _by_seed(v) for k, v in critic_sets.items()}
    if ref not in sets:
        raise KeyError(f"ref {ref!r} is not one of {list(sets)}")
    seeds = sorted(set.intersection(*[set(v) for v in sets.values()]), key=str)
    skipped = set.union(*[set(v) for v in sets.values()]) - set(seeds)
    if not seeds:
        raise KeyError("no seed is present in all critic sets")
    if skipped:
        warnings.warn(f"seeds {sorted(skipped, key=str)} are not in all critic sets and were skipped")

    first = None
    curves, shapes, scales = [], [], []
    for seed in seeds:
        m_ref = sets[ref][seed]
        p = store.processor(m_ref, split)
        s, a_clin, hs = D.states(p), D.actions(p), D.health_score(p)[0]
        if first is None:
            bidx, labels = severity_bins(hs, n_bins)
            nb = len(labels)
            x = a_clin[:, dim]
            support = np.zeros(nb)
            for b in range(nb):
                nz = x[(bidx == b) & (x > D.NONZERO)]
                support[b] = np.quantile(nz, support_q) if len(nz) else 0.0
            if support.max() <= 0:
                raise ValueError(f"no nonzero {DIM_NAMES[dim]} doses in {split}")
            grid = np.linspace(0.0, float(support.max()), n_grid)
            first = (a_clin, hs)
        elif (a_clin.shape != first[0].shape or not np.allclose(a_clin, first[0])
              or not np.allclose(hs, first[1], equal_nan=True)):
            raise ValueError(f"seed {seed}: clinician actions or health score differ from the first seed's; "
                             f"the seeds are not row-aligned")
        for name, sset in sets.items():
            m = sset[seed]
            if name != ref:
                check_judge_matches(m_ref, m)
            q_clin = m.q(s, a_clin, reduction=reduction, batch_size=batch_size)
            iqr = max(float(np.quantile(q_clin, 0.75) - np.quantile(q_clin, 0.25)), 1e-6)
            q = np.empty((len(s), n_grid))
            a = a_clin.copy()
            for j, d in enumerate(grid):
                a[:, dim] = d
                q[:, j] = m.q(s, a, reduction=reduction, batch_size=batch_size)
            delta = (q - q[:, :1]) / iqr
            scales.append({"critic": name, "seed": seed, "iqr": iqr})
            for b in range(nb):
                rows = bidx == b
                sup = grid <= support[b] + 1e-12                 # a prefix of the grid
                mean_curve = delta[rows].mean(0) if rows.any() else np.full(n_grid, np.nan)
                for j in range(n_grid):
                    curves.append({"critic": name, "seed": seed, "bin": b, "dose": grid[j],
                                   "delta": mean_curve[j], "supported": bool(sup[j])})
                row = {"critic": name, "seed": seed, "bin": b, "n": int(rows.sum()),
                       **{k: np.nan for k in SHAPES}, "delta_top": np.nan}
                if rows.any() and sup.sum() >= 2:
                    c = delta[rows][:, sup]
                    cls = _classify(c, flat_tol)
                    row.update({k: float((cls == i).mean()) for i, k in enumerate(SHAPES)})
                    row["delta_top"] = float(c[:, -1].mean())
                shapes.append(row)
        if verbose:
            print(f"  seed {seed}: {len(s)} rows, {len(sets)} critics", flush=True)

    settings = dict(n_bins=n_bins, n_grid=n_grid, support_q=support_q, flat_tol=flat_tol, reduction=reduction)
    return DoseResponse(pd.DataFrame(curves), pd.DataFrame(shapes), labels, support, grid, pd.DataFrame(scales),
                        dim, split, settings)


def summarize_shapes(res: DoseResponse) -> pd.DataFrame:
    """Per critic and severity bin: n rows, then mean / min / max across seeds of the shape fractions and delta_top."""
    cols = list(SHAPES) + ["delta_top"]
    agg = {"n": "first", **{k: ["mean", "min", "max"] for k in cols}}
    out = res.shapes.groupby(["critic", "bin"]).agg(agg)
    out.insert(0, ("health_score", "range"), [res.labels[b] for b in out.index.get_level_values("bin")])
    return out


# =============================================================================
# Observed outcomes by dose (model-free reference)
# =============================================================================

def _stratified_diffs(G, stay, fine, group, n_stays, n_bins, n_fine, n_groups, scale, n_boot=500, boot_seed=0):
    """
    Difference in mean observed return between each nonzero dose group and the no-fluid group, within severity bins,
    compared inside fine health-score strata and averaged with the dose group's row counts as weights.

    G, stay, fine, group : one entry per row used: observed return, stay index (0 .. n_stays-1), fine health-score
                           stratum (0 .. n_bins * n_fine - 1, nested in the bins) and dose group (0 = no fluid)
    Returns (point, draws, raw), each divided by `scale`:
      point (n_bins, n_groups)          adjusted difference (column k = dose group k + 1)
      draws (n_boot, n_bins, n_groups)  the same on bootstrap resamples of the stays
      raw   (n_bins, n_groups)          unadjusted difference of the bin's group means (shows the confounding)
    NaN where a bin lacks rows of the group or of the no-fluid group.
    """
    F, K = n_bins * n_fine, n_groups + 1
    key = (np.asarray(stay) * F + fine) * K + group
    S = np.bincount(key, weights=G, minlength=n_stays * F * K).reshape(n_stays, F, K)
    C = np.bincount(key, minlength=n_stays * F * K).reshape(n_stays, F, K).astype(np.float64)
    rng = np.random.default_rng(boot_seed)
    W = np.vstack([np.ones((1, n_stays)), rng.multinomial(n_stays, np.full(n_stays, 1.0 / n_stays), size=n_boot)])
    St, Ct = np.einsum("bj,jfk->bfk", W, S), np.einsum("bj,jfk->bfk", W, C)      # row 0 = the data itself
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (St / Ct)[:, :, 1:] - (St / Ct)[:, :, :1]                             # (B+1, F, n_groups)
        ok = np.isfinite(diff)
        w = np.where(ok, Ct[:, :, 1:], 0.0).reshape(-1, n_bins, n_fine, n_groups)
        adj = (np.where(ok, diff, 0.0).reshape(-1, n_bins, n_fine, n_groups) * w).sum(2) / w.sum(2)
        Sc, Cc = (x.reshape(-1, n_bins, n_fine, K).sum(2) for x in (St, Ct))
        raw = (Sc / Cc)[:, :, 1:] - (Sc / Cc)[:, :, :1]
    return adj[0] / scale, adj[1:] / scale, raw[0] / scale


def observed_by_dose(res: DoseResponse, critic_sets: Dict[str, Dict], store, names: List[str], ref: str,
                     n_groups: int = 3, n_fine: int = 4, n_boot: int = 500, boot_seed: int = 0,
                     tail_weight: float = 0.05, verbose: bool = True) -> pd.DataFrame:
    """
    Observed-outcome reference for the critics in `names` (SARSA critics: a CQL critic follows the policy after the
    first action, so the clinician data are not its reference). `res`, `critic_sets` and `ref` are those of
    dose_response; every critic's own reward, discount and reward_scaling are used for its observed return.

    One row per (critic, seed, bin, dose group): dose (mean dose of the group), n (rows), n_zero (no-fluid rows in
    the bin), diff / lo / hi (adjusted difference to the no-fluid group in the critic's IQR units, 95% stay-bootstrap
    interval) and raw_diff (unadjusted). The units are those of the curves from dose_response, so they can be
    overlaid (plot_dose_response(res, observed=obs)).
    """
    sets = {k: _by_seed(v) for k, v in critic_sets.items()}
    dim, split, nb = res.dim, res.split, len(res.labels)
    out = []
    for seed in sorted(res.curves["seed"].unique()):
        p_ref = store.processor(sets[ref][seed], split)
        hs, x = D.health_score(p_ref)[0], D.actions(p_ref)[:, dim]
        traj = D.trajectory_ids(p_ref)
        fine, lab_f = severity_bins(hs, nb * n_fine)
        coarse, lab_c = severity_bins(hs, nb)
        if len(lab_f) != nb * n_fine or len(lab_c) != nb:
            raise ValueError("health-score quantile edges are tied; use fewer bins or fine strata")
        valid = (fine >= 0) & (coarse >= 0)
        if not np.array_equal(fine[valid] // n_fine, coarse[valid]):
            raise ValueError("fine health-score strata are not nested in the severity bins")
        inv = np.unique(traj, return_inverse=True)[1]
        n_stays = int(inv.max()) + 1
        supported = x <= np.where(valid, res.support[np.maximum(coarse, 0)], -1.0) + 1e-12
        for name in names:
            m = sets[name][seed]
            p = store.processor(m, split)
            if name != ref and (D.states(p).shape != D.states(p_ref).shape or not np.allclose(D.states(p), D.states(p_ref))
                                or not np.allclose(D.actions(p), D.actions(p_ref))
                                or not np.array_equal(D.trajectory_ids(p), traj)):
                raise ValueError(f"{m.record.label}: rows, states or actions differ from {ref}'s; not row-aligned")
            gamma = m.record.get("discount")
            if gamma is None:
                raise KeyError(f"{m.record.label}: no 'discount' in the run's hyperparameters")
            rs = m.record.get("reward_scaling")
            G = discounted_return_to_go(D.rewards(p) * (1.0 if rs is None else float(rs)), traj, float(gamma))
            use = valid & supported & (rows_after_in_stay(traj) >= interior_horizon(float(gamma), tail_weight))
            nonzero = x > D.NONZERO
            edges = np.quantile(x[use & nonzero], np.linspace(0, 1, n_groups + 1)[1:-1])
            group = np.where(nonzero, 1 + np.digitize(x, edges), 0)
            iqr = float(res.scales[(res.scales.critic == name) & (res.scales.seed == seed)]["iqr"].iloc[0])
            point, draws, raw = _stratified_diffs(G[use], inv[use], fine[use], group[use], n_stays, nb, n_fine,
                                                  n_groups, iqr, n_boot=n_boot, boot_seed=boot_seed)
            for b in range(nb):
                in_bin = use & (coarse == b)
                for k in range(1, n_groups + 1):
                    g = in_bin & (group == k)
                    d = draws[:, b, k - 1]
                    lo, hi = np.nanpercentile(d, [2.5, 97.5]) if np.isfinite(d).any() else (np.nan, np.nan)
                    out.append({"critic": name, "seed": seed, "bin": b, "group": k,
                                "dose": float(x[g].mean()) if g.any() else np.nan, "n": int(g.sum()),
                                "n_zero": int((in_bin & (group == 0)).sum()),
                                "diff": point[b, k - 1], "lo": lo, "hi": hi, "raw_diff": raw[b, k - 1]})
        if verbose:
            print(f"  seed {seed}: observed returns for {len(names)} critics", flush=True)
    return pd.DataFrame(out)


def summarize_observed(obs: pd.DataFrame) -> pd.DataFrame:
    """Per critic, severity bin and dose group: mean across seeds of dose, rows and the (adjusted and raw) differences
    with their intervals, and the min / max of the adjusted difference across seeds."""
    g = obs.groupby(["critic", "bin", "group"])
    out = g[["dose", "n", "n_zero", "diff", "lo", "hi", "raw_diff"]].mean()
    out["diff_min"], out["diff_max"] = g["diff"].min(), g["diff"].max()
    return out


# =============================================================================
# Plots
# =============================================================================

def _bin_title(res: DoseResponse, b: int) -> str:
    tag = " (sickest)" if b == 0 else " (healthiest)" if b == len(res.labels) - 1 else ""
    return f"health score {res.labels[b]}{tag}"


def plot_dose_response(res: DoseResponse, critics: Optional[List[str]] = None, observed: Optional[pd.DataFrame] = None,
                       save_path=None):
    """
    Curves of delta against dose: one row per critic, one column per severity bin (sickest left). Line = mean
    across seeds, band = min-max across seeds, supported doses only. All panels share the y axis (IQR units),
    so shapes can be compared across critics and bins.

    observed: the output of observed_by_dose. For the critics it contains, the observed difference to the no-fluid
    group is drawn as open circles at each dose group's mean dose (mean across seeds, error bars = mean 95% stay-
    bootstrap interval), in the same units as the curve.
    """
    critics = list(critics or res.curves["critic"].unique())
    if len(critics) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} critics per figure")
    nb = len(res.labels)
    n_rows = res.shapes[res.shapes.critic == critics[0]].groupby("bin")["n"].first()
    fig, axes = plt.subplots(len(critics), nb, figsize=(2.9 * nb, 2.5 * len(critics)),
                             sharex=True, sharey=True, squeeze=False)
    cur = res.curves[res.curves.supported]
    legend_done = False
    for i, name in enumerate(critics):
        color = PALETTE[i]
        for b in range(nb):
            ax = axes[i, b]
            piv = cur[(cur.critic == name) & (cur.bin == b)].pivot(index="dose", columns="seed", values="delta")
            ax.fill_between(piv.index, piv.min(1), piv.max(1), color=color, alpha=0.18, linewidth=0)
            ax.plot(piv.index, piv.mean(1), color=color, linewidth=2, label="critic")
            ax.axhline(0, color=GREY, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if observed is not None and name in set(observed["critic"]):
                o = observed[(observed.critic == name) & (observed["bin"] == b)].groupby("group")[
                    ["dose", "diff", "lo", "hi"]].mean()
                ax.errorbar(o["dose"], o["diff"], yerr=[(o["diff"] - o["lo"]).clip(lower=0), (o["hi"] - o["diff"]).clip(lower=0)],
                            fmt="o", color=color, mfc="white", ms=5, capsize=2.5, linewidth=1.2, label="observed")
                if b == 0 and not legend_done:
                    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
                    legend_done = True
            if i == 0:
                ax.set_title(f"{_bin_title(res, b)}\nn = {int(n_rows[b])}", fontsize=8.5)
            if b == 0:
                ax.set_ylabel(f"{name}\nQ change vs none (IQR units)", fontsize=8.5)
            if i == len(critics) - 1:
                ax.set_xlabel(f"{DIM_NAMES[res.dim]} dose", fontsize=9)
    n_seeds = res.curves["seed"].nunique()
    fig.suptitle(f"Critic response to {DIM_NAMES[res.dim]} dose by severity, {res.split} set "
                 f"(line = mean of {n_seeds} seeds, band = min-max)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_shape_fractions(res: DoseResponse, critics: Optional[List[str]] = None, save_path=None):
    """Stacked bars: share of rows in each severity bin whose supported curve is lower / interior / higher /
    irregular / flat (mean across seeds), one panel per critic."""
    critics = list(critics or res.shapes["critic"].unique())
    nb = len(res.labels)
    summ = res.shapes.groupby(["critic", "bin"])[list(SHAPES)].mean()
    fig, axes = plt.subplots(1, len(critics), figsize=(3.6 * len(critics), 3.4), sharey=True, squeeze=False)
    x = np.arange(nb)
    ticks = [str(i + 1) for i in range(nb)]
    ticks[0] += "\n(sickest)"
    if nb > 1:
        ticks[-1] += "\n(healthiest)"
    for ax, name in zip(axes[0], critics):
        bottom = np.zeros(nb)
        for k in SHAPES:
            v = summ.loc[name][k].reindex(range(nb)).fillna(0).values * 100
            ax.bar(x, v, bottom=bottom, color=SHAPE_COLORS[k], edgecolor="white", linewidth=1.5, label=k)
            bottom += v
        ax.set_title(name, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_xlabel("health-score bin", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("% of rows", fontsize=9)
    axes[0, -1].legend(frameon=False, fontsize=8, title="curve shape", title_fontsize=8, loc="upper left",
                       bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"Shape of the {DIM_NAMES[res.dim]} dose-response curve, {res.split} set", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig