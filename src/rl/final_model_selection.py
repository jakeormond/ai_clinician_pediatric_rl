"""
Final-model selection pipeline, stage 2 -> stage 3, for a single reward
type's alpha sweep (call once for health-score, once for terminal).

Two-stage filter, both criteria decided BEFORE looking at which alpha they
happen to favor:
  1. Exclude any alpha with own-critic advantage <= 0 -- direct evidence
     the actor failed to optimize against its own training signal,
     confirmed (via the stochastic-vs-deterministic check) to be a real
     training issue at very high alpha, not an eval artifact.
  2. Among the survivors, find the smallest alpha at which SARSA-advantage
     becomes UNSTABLE (a precise, sliding-window criterion, not eyeballed)
     and keep only alphas strictly below that onset point.
"""

import numpy as np


def filter_by_own_critic_advantage(results, normalized=True, min_advantage=0.0):
    """
    results: dict {alpha: {...}} as returned by run_alpha_sweep_analysis.
    Returns (eligible_alphas, excluded_alphas), both sorted ascending.
    """
    key = "own_advantage_mean_normalized" if normalized else "own_advantage_mean"
    alphas = sorted(results.keys())

    eligible = [a for a in alphas if results[a][key] > min_advantage]
    excluded = [a for a in alphas if results[a][key] <= min_advantage]

    print(f"Own-critic advantage filter (threshold > {min_advantage}, "
          f"{'normalized' if normalized else 'raw'}):")
    print(f"  Eligible ({len(eligible)}): {eligible}")
    print(f"  Excluded ({len(excluded)}): {excluded}")
    for a in excluded:
        print(f"    alpha={a}: own-critic advantage = {results[a][key]:.4f}")

    return eligible, excluded


def find_sarsa_instability_onset(
        results, eligible_alphas, normalized=True,
        window=3, relative_threshold=0.5,
    ):
    """
    Detects the smallest alpha at which SARSA-advantage becomes unstable,
    using a sliding-window criterion: for each alpha (with enough
    neighbors on either side to fill `window`), compute the standard
    deviation of SARSA-advantage (mean across SARSA seeds) within that
    window of adjacent alphas. Flag instability where this local std
    exceeds `relative_threshold` * the local window mean (in absolute
    value) -- i.e. the window's spread is large relative to its own level,
    a scale-free way to say "this region is jumping around a lot".

    eligible_alphas: alphas that passed the own-critic-advantage filter --
        instability is only assessed among these, in ascending order.
    window: number of consecutive (eligible) alpha points per window.
    relative_threshold: local_std / |local_mean| above this flags instability.

    Returns:
        onset_alpha: the smallest alpha where instability was first
            detected (None if never detected across the eligible range).
        stable_alphas: eligible alphas strictly below onset_alpha (all of
            them, if onset_alpha is None).
        diagnostics: per-window (center_alpha, local_mean, local_std, ratio)
            for inspection/plotting.
    """
    suffix = "_normalized" if normalized else ""
    alphas = sorted(eligible_alphas)
    sarsa_vals = np.array([results[a][f"sarsa_advantage_means{suffix}"].mean() for a in alphas])

    diagnostics = []
    onset_alpha = None

    for i in range(len(alphas) - window + 1):
        window_alphas = alphas[i:i + window]
        window_vals = sarsa_vals[i:i + window]
        local_mean = window_vals.mean()
        local_std = window_vals.std()
        ratio = local_std / (abs(local_mean) + 1e-8)

        center_alpha = window_alphas[len(window_alphas) // 2]
        diagnostics.append({
            "window_alphas": window_alphas, "center_alpha": center_alpha,
            "local_mean": local_mean, "local_std": local_std, "ratio": ratio,
        })

        if ratio > relative_threshold and onset_alpha is None:
            onset_alpha = window_alphas[0]  # first alpha in the first unstable window

    if onset_alpha is not None:
        stable_alphas = [a for a in alphas if a < onset_alpha]
    else:
        stable_alphas = list(alphas)

    print(f"\nSARSA-advantage instability scan (window={window}, "
          f"relative_threshold={relative_threshold}):")
    for d in diagnostics:
        flag = " <-- UNSTABLE" if d["ratio"] > relative_threshold else ""
        print(f"  window {d['window_alphas']}: mean={d['local_mean']:.4f}, "
              f"std={d['local_std']:.4f}, ratio={d['ratio']:.2f}{flag}")

    if onset_alpha is not None:
        print(f"\nInstability onset detected at alpha={onset_alpha}.")
        print(f"Stable alphas (kept): {stable_alphas}")
    else:
        print("\nNo instability detected across the eligible range -- all eligible alphas kept.")

    return onset_alpha, stable_alphas, diagnostics


def select_final_candidates(results, normalized=True, min_advantage=0.0, window=3, relative_threshold=0.5,
                             safety_margin=1):
    """
    Runs the full two-stage selection and returns the final list of
    candidate alphas (own-critic-filtered, then SARSA-instability-filtered),
    ready for a final pick (e.g., by highest SARSA-advantage among these,
    or by occupancy support) and multi-seed confirmation.

    safety_margin: additionally drops this many of the LARGEST stable
        alphas beyond the detected instability onset, as an extra buffer --
        the boundary alpha closest to instability is the one least trusted,
        since its own neighbors are already flagged as unreliable.
        Default 1 (drop the single largest stable alpha in addition to
        everything at/above the detected onset).
    """
    eligible, excluded = filter_by_own_critic_advantage(results, normalized=normalized, min_advantage=min_advantage)
    onset_alpha, stable_alphas, diagnostics = find_sarsa_instability_onset(
        results, eligible, normalized=normalized, window=window, relative_threshold=relative_threshold
    )

    if safety_margin > 0 and len(stable_alphas) > safety_margin:
        dropped = stable_alphas[-safety_margin:]
        stable_alphas = stable_alphas[:-safety_margin]
        print(f"\nSafety margin: additionally dropped {dropped} "
              f"(closest to detected instability onset).")
        print(f"Final stable alphas: {stable_alphas}")

    return stable_alphas, {"excluded_by_own_critic": excluded, "instability_onset": onset_alpha, "diagnostics": diagnostics}


def rank_stable_candidates(results, stable_alphas, normalized=True, rank_by="occupancy"):
    """
    Ranks the stable, eligible alphas, printing both SARSA-advantage and
    occupancy support for context regardless of ranking choice.

    rank_by:
        "occupancy" (default) -- ranks by occupancy support descending,
            SARSA-advantage as tie-break. This is the SAFETY-FIRST choice:
            since occupancy rises roughly monotonically with alpha within
            the stable region, this generally selects the most
            conservative model that still passed both stability filters,
            rather than the one with peak (but potentially still-risky,
            e.g. residual extreme-dose) advantage.
        "sarsa_advantage" -- ranks by SARSA-advantage descending,
            occupancy as tie-break (the original, advantage-first ranking).

    Returns the sorted list of (alpha, sarsa_advantage, occupancy)
    tuples, best first under the chosen criterion.
    """
    suffix = "_normalized" if normalized else ""
    candidates = []
    for a in stable_alphas:
        sarsa_adv = results[a][f"sarsa_advantage_means{suffix}"].mean()
        occ = results[a]["occupancy_support_frac"]
        candidates.append((a, sarsa_adv, occ))

    if rank_by == "occupancy":
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    elif rank_by == "sarsa_advantage":
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    else:
        raise ValueError(f"Unknown rank_by='{rank_by}', use 'occupancy' or 'sarsa_advantage'")

    print(f"Stable candidates ranked by {rank_by} (descending, other metric as tie-break):")
    for alpha, sarsa_adv, occ in candidates:
        print(f"  alpha={alpha}: occupancy={occ*100:.1f}%, SARSA-advantage={sarsa_adv:.4f}")

    print(f"\nTop candidate: alpha={candidates[0][0]}")
    return candidates

if __name__ == "__main__":
    print("=== Health-score reward ===")
    stable_alphas_hs, info_hs = select_final_candidates(results_health_score)

    print("\n=== Terminal reward ===")
    stable_alphas_term, info_term = select_final_candidates(results_terminal)