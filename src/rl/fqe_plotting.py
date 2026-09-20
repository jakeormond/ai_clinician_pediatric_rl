"""
Plotting functions for the FQE (Fitted Q Evaluation) analysis.

Consumes the output of evaluate_fqe_final_seeds.run_matched_seed_fqe_advantage
(a list of per-seed result dicts with keys 'seed', 'advantage',
'advantage_normalized', 'q_clinician', 'q_cql', 'q_iqr') -- add further FQE
figures to this file as the analysis grows, rather than to
evaluate_fqe_final_seeds.py, which is for computation only.
"""

import numpy as np
import matplotlib.pyplot as plt


def normal_approx_ci(values, ci=95):
    """
    Analytic mean +/- z*SEM interval. With the thousands of (largely
    independent-ish) per-state values here, the bootstrap distribution of
    the mean is already well into its normal limit, so this matches a full
    bootstrap to within noise at a fraction of the cost -- no resampling,
    O(n) instead of O(n_boot * n).
    """
    from scipy import stats
    values = np.asarray(values)
    n = len(values)
    sem = values.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(1 - (100 - ci) / 200)
    m = values.mean()
    return m - z * sem, m + z * sem


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0, chunk_size=200):
    """
    Kept for cases where you actually want resampling (e.g. sanity-checking
    normal_approx_ci against a skewed/small sample) rather than the fast
    analytic version used by default in plot_fqe_advantage_summary.
    Processes in chunks so memory stays at chunk_size * n rather than
    n_boot * n.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boot_means = np.empty(n_boot)
    for start in range(0, n_boot, chunk_size):
        b = min(chunk_size, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        boot_means[start:start + b] = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


def episode_ids_from_dones(dones):
    """
    Builds a per-state episode/stay id array from a 'dones' array (1 marks
    the last transition of an episode, matching processor.rl_dataset's
    convention: rows are (s_t, a_t, r_t, s_{t+1}, done_t)). Consecutive
    rows get the same id until a done=1 row; the next row starts a new id.
    Same length as `dones`, safe to pass straight into block_bootstrap_ci.
    """
    dones = np.asarray(dones).astype(int)
    n = len(dones)
    episode_ids = np.zeros(n, dtype=int)
    if n > 1:
        episode_ids[1:] = np.cumsum(dones[:-1])
    return episode_ids


def block_bootstrap_ci(values, group_ids, n_boot=2000, ci=95, seed=0):
    """
    Bootstrap CI that resamples whole trajectories (e.g. patient stays)
    rather than individual per-state rows. Per-state advantage values from
    the same stay are serially correlated (same underlying patient, same
    critic, adjacent hourly states), so resampling rows independently (as
    normal_approx_ci and bootstrap_ci both do) understates the true
    uncertainty. This resamples groups with replacement instead, which is
    the honest unit of independence here.

    values: 1D array of per-state values (e.g. one seed's advantage_normalized).
    group_ids: 1D array, same length as values, giving each state's
        trajectory/stay identifier -- build this once via
        episode_ids_from_dones(processors['test'].rl_dataset["dones"]) and
        reuse it across all 5 seeds, since they share the same test set.
    """
    values = np.asarray(values)
    group_ids = np.asarray(group_ids)
    if len(group_ids) != len(values):
        raise ValueError(f"group_ids length ({len(group_ids)}) must match values length ({len(values)})")

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    # index lookup: which row indices belong to each group
    group_to_rows = {g: np.where(group_ids == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        rows = np.concatenate([group_to_rows[g] for g in sampled_groups])
        boot_means[b] = values[rows].mean()

    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


def sign_test(seed_means):
    """
    Exact one-sided sign test: under the null that a seed's mean advantage
    is equally likely to be positive or negative, tests whether the
    observed number of positive seeds is more than chance would produce.
    Nearly assumption-free (doesn't rely on the per-state distribution at
    all) -- appropriate given n=5 is too small to lean on a normal
    approximation.

    Returns (n_positive, n_total, p_value).
    """
    from scipy import stats
    seed_means = np.asarray(seed_means)
    n_positive = int((seed_means > 0).sum())
    n_total = len(seed_means)
    result = stats.binomtest(n_positive, n_total, p=0.5, alternative="greater")
    return n_positive, n_total, result.pvalue


def across_seed_ci(seed_means, ci=95):
    """
    Mean +/- t*SEM across the seed-level means themselves (n=5, not
    n=states). This is the correct unit of replication for "does training
    a CQL policy this way tend to beat the clinician" -- each seed is an
    independent training run, unlike the per-state values within one seed.
    Uses a t- rather than normal-distribution critical value since n is
    small.
    """
    from scipy import stats
    seed_means = np.asarray(seed_means)
    n = len(seed_means)
    mean = seed_means.mean()
    sem = seed_means.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - (100 - ci) / 200, df=n - 1)
    return mean - t * sem, mean + t * sem


def fraction_positive(values):
    """
    Fraction of per-state advantage values that are > 0. More robust than
    the mean to the small number of extreme states that dominate a
    heavy-tailed distribution's average -- answers "for what fraction of
    decisions did the policy's action score at least as well as the
    clinician's," independent of how large any single advantage was.
    """
    values = np.asarray(values)
    return (values > 0).mean()


def summarize_fqe_results(results, key="advantage_normalized"):
    """
    Prints and returns the seed-level summary statistics that should back
    an "FQE indicates the trained policy beats the clinician" claim:
      - per-seed mean and fraction-positive
      - sign test across the 5 seed-level means
      - t-based 95% CI across the 5 seed-level means (the right level of
        replication for this claim -- see across_seed_ci)
    Does not include the block-bootstrap CI, since that needs per-state
    group/stay ids that aren't in `results` -- call block_bootstrap_ci
    directly per seed once you have those.
    """
    results = sorted(results, key=lambda r: r["seed"])
    seed_means = np.array([r[key].mean() for r in results])
    frac_pos = np.array([fraction_positive(r[key]) for r in results])

    n_positive, n_total, p_value = sign_test(seed_means)
    ci_lo, ci_hi = across_seed_ci(seed_means)

    print("Per-seed summary:")
    for r, m, fp in zip(results, seed_means, frac_pos):
        print(f"  seed {r['seed']}: mean={m:.4f}, fraction of states positive={fp:.3f}")

    print(f"\nSign test: {n_positive}/{n_total} seeds positive, "
          f"one-sided p={p_value:.4f} (H0: each seed equally likely +/-)")
    print(f"Across-seed mean (n={n_total}): {seed_means.mean():.4f}, "
          f"95% CI (t-based): [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "seed_means": seed_means,
        "fraction_positive": frac_pos,
        "sign_test_n_positive": n_positive,
        "sign_test_n_total": n_total,
        "sign_test_p_value": p_value,
        "across_seed_ci": (ci_lo, ci_hi),
    }


def choose_linthresh(results, key="advantage_normalized", pct=25):
    """
    Picks a linear threshold for a symlog axis from the data itself: the
    `pct`-th percentile of |value| across all seeds, excluding exact zeros.
    Below this, values are shown on a linear scale (so the bulk of the mass
    near zero doesn't collapse to a line); above it, symlog takes over so
    the long tails don't dominate the axis range.
    """
    all_vals = np.concatenate([np.abs(r[key]) for r in results])
    nonzero = all_vals[all_vals > 0]
    return np.percentile(nonzero, pct)


def plot_fqe_advantage_summary(results, save_prefix="fqe_advantage_summary", linthresh=None,
                                 group_ids=None, n_boot=2000):
    """
    Two-panel summary of the matched-seed FQE advantage results:
      A) per-state distribution of normalized advantage, one violin per seed,
         on a symlog y-axis (linear near zero, log beyond +/-linthresh) --
         plain linear or percentile-clipped axes are unusable here since the
         per-state advantage is heavy-tailed in both directions relative to
         its mean, which otherwise collapses the bulk of the distribution to
         a line at zero
      B) mean normalized advantage per seed with a 95% CI, linear scale --
         means and their CIs sit close together so log scaling isn't needed
         here

    results: the list of dicts returned by eval_fqe.run_matched_seed_fqe_advantage.
    linthresh: linear-region half-width for the symlog axis in panel A. If
        None, picked automatically from the data via choose_linthresh --
        pass an explicit value if you want it stable across figures/re-runs.
    group_ids: per-state trajectory/stay id array (e.g. from
        episode_ids_from_dones), shared across all seeds since they're
        computed over the same test set. If given, panel B uses
        block_bootstrap_ci (correlation-aware, wider, honest); if omitted,
        falls back to normal_approx_ci, which is faster but overstates
        precision by treating serially-correlated per-state rows as
        independent -- prefer passing group_ids whenever you have it.
    n_boot: bootstrap resamples per seed, only used when group_ids is given.
    """
    results = sorted(results, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in results]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(seeds))]

    if linthresh is None:
        linthresh = choose_linthresh(results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})

    # --- Panel A: per-state distribution, symlog y-axis ---
    ax = axes[0]
    data_full = [r["advantage_normalized"] for r in results]

    parts = ax.violinplot(data_full, positions=seeds, showextrema=False, widths=0.7)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
        pc.set_edgecolor("none")

    medians = [np.median(d) for d in data_full]
    ax.scatter(seeds, medians, color=colors, zorder=3, s=30, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel("CQL / FQE seed")
    ax.set_ylabel("Normalized advantage\n" r"$Q_{FQE}(\pi_{CQL}) - Q_{FQE}(a_{clinician})$")
    ax.set_xticks(seeds)
    ax.set_title(f"Per-state distribution (symlog, linthresh={linthresh:.1e})")

    # --- Panel B: mean +/- 95% CI ---
    ax = axes[1]
    means = [r["advantage_normalized"].mean() for r in results]
    if group_ids is not None:
        cis = [block_bootstrap_ci(r["advantage_normalized"], group_ids, n_boot=n_boot)
               for r in results]
        ci_label = "block-bootstrap"
    else:
        cis = [normal_approx_ci(r["advantage_normalized"]) for r in results]
        ci_label = "normal approx."
    yerr_lo = [m - lo for m, (lo, hi) in zip(means, cis)]
    yerr_hi = [hi - m for m, (lo, hi) in zip(means, cis)]

    ax.errorbar(seeds, means, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=4, zorder=2)
    ax.scatter(seeds, means, color=colors, s=60, zorder=3, edgecolor="black", linewidth=0.6)
    ax.axhline(np.mean(means), color="grey", linewidth=1, linestyle=":", label="mean across seeds")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("CQL / FQE seed")
    ax.set_ylabel("Mean normalized advantage")
    ax.set_xticks(seeds)
    ax.set_title(f"Mean advantage \u00b1 95% CI ({ci_label})")

    fig.suptitle(r"FQE-estimated advantage of $\pi_{CQL}$ over logged clinician actions",
                 y=1.03, fontsize=12)
    fig.tight_layout()

    fig.savefig(f"{save_prefix}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    return fig, axes


# Usage in the notebook, right after `results = eval_fqe.run_matched_seed_fqe_advantage(...)`:
#
#   import src.rl.fqe_plotting as fqe_plotting
#
#   stay_ids_test = fqe_plotting.episode_ids_from_dones(processors['test'].rl_dataset["dones"])
#
#   fig, axes = fqe_plotting.plot_fqe_advantage_summary(results, group_ids=stay_ids_test)
#   plt.show()
#
#   summary = fqe_plotting.summarize_fqe_results(results)
#
#   for r in results:
#       lo, hi = fqe_plotting.block_bootstrap_ci(r["advantage_normalized"], stay_ids_test)
#       print(f"seed {r['seed']}: block-bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")