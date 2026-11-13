"""
src/rl/evaluate_doubly_robust.py

Doubly robust (DR) off-policy evaluation of the CQL policies on MIMIC: the per-decision importance weights of
evaluate_wis.py combined with the FQE critic of evaluate_fqe_final_seeds.py. Part of the WIS analysis notebook.

Why. The weights alone are unreliable here: the policy puts vasopressor doses far from the clinicians' (policy median
0.13 vs 0.345) and has a heavy fluid-dose tail, so a few stays carry the whole stepwise estimate and ESS collapses
above a weight cap of ~50. The critic alone (the FQE estimate V_pi) has no such problem but carries the critic's own
bias. DR uses the critic as the main estimate and the weights only to correct the critic's errors on the logged data:
it is accurate if EITHER the critic OR the weights are, and its variance is small when the critic is good.

The estimator (weighted per-decision DR, Thomas & Brunskill 2016; Jiang & Li 2016). With w_t the cumulative
importance weight through hour t, Qhat = the FQE critic of the target policy and Vhat(s) = Qhat(s, pi(s)):

    V_DR = sum_t gamma^t [ sum_i w_t^i (r_t^i - Qhat(s_t^i, a_t^i)) / sum_i w_t^i
                           + sum_i w_{t-1}^i Vhat(s_t^i) / sum_i w_{t-1}^i ]          (w_{-1} = 1)

the sums running over the stays still active at hour t, with the same convention and the same upper-tail cap on the
cross-sectional weights as compute_stepwise_wis. At t = 0 the second term is the plain critic estimate mean Vhat(s_0).

The matching null. With all log-ratios 0 and Vhat(s) = Qhat(s, a_logged) (the policy IS the clinician) the two critic
terms cancel and V_DR is exactly the stepwise null (mean reward among active stays, discounted), whatever the critic.
So `dr_minus_null` compares like with like and is directly comparable to the stepwise "policy - null" of the WIS tables.
It is also checked numerically against compute_stepwise_wis on every run.

Units. Rewards are the processor's own (as in the WIS tables). The critic returns health-score points
(scaled=True); multiplying by the critic's reward_scale gives them in reward units.

Caveats. (1) Vhat uses the policy's deterministic action while the weights use its stochastic density; the DR
identity is then approximate. (2) The critic is the FQE of the SAME seed; the pairing is checked. (3) Bootstrap CIs
resample stays (fixed critic and weights): they cover patient sampling, not critic-training variability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Core estimator (pure numpy)
# =============================================================================

def dense_matrices(log_ratio, rewards, q_logged, v_hat, lengths):
    """Stack per-stay lists of arrays into (n_stays, max_len) matrices (zero padded; `lengths` says where to stop)."""
    n, T = len(lengths), int(max(lengths))
    out = [np.zeros((n, T)) for _ in range(4)]
    for i, (lr, r, q, v) in enumerate(zip(log_ratio, rewards, q_logged, v_hat)):
        L = len(r)
        out[0][i, :L], out[1][i, :L], out[2][i, :L], out[3][i, :L] = lr, r, q, v
    return out + [np.asarray(lengths, dtype=np.int64)]


def _weights(lr, cap):
    """Self-normalised weights from log-weights, upper tail capped at the `cap`-th percentile (log space)."""
    if cap is not None and len(lr) > 1:
        lr = np.minimum(lr, np.percentile(lr, cap))
    w = np.exp(lr - lr.max())
    return w / w.sum()


def dr_estimate(LR, R, Q, V, L, gamma, cap=None, idx=None):
    """
    Weighted per-decision DR on the dense matrices (see module docstring). idx = stay indices of a bootstrap resample.
    Returns a dict:
      dr        doubly robust estimate
      step      stepwise WIS alone (importance weights, no critic) -- equals compute_stepwise_wis
      dm        critic alone: mean Vhat(s_0)
      null      the stepwise null: same estimator with log-ratio 0 (the clinician) -- subtract it from dr and step
      dr_null   DR with log-ratio 0 and Vhat = Qhat(s, a_logged); must equal `null` exactly (internal check)
    """
    if idx is not None:
        LR, R, Q, V, L = LR[idx], R[idx], Q[idx], V[idx], L[idx]
    cum = np.cumsum(LR, axis=1)
    prev = np.concatenate([np.zeros((len(L), 1)), cum[:, :-1]], axis=1)
    dr = step = null = dr_null = 0.0
    for t in range(int(L.max())):
        a = L > t
        r, q, v = R[a, t], Q[a, t], V[a, t]
        w, wp = _weights(cum[a, t], cap), _weights(prev[a, t], cap)
        g = gamma ** t
        dr += g * (w @ (r - q) + wp @ v)
        step += g * (w @ r)
        null += g * r.mean()
        dr_null += g * ((r - q).mean() + q.mean())
        if t == 0:
            dm = float(v.mean())
    return {"dr": dr, "step": step, "dm": dm, "null": null, "dr_null": dr_null}


# =============================================================================
# Per-seed driver
# =============================================================================

def _critic_values(cql, fqe, processor, batch_size=4096):
    """Qhat(s, a_logged) and Vhat(s) = Qhat(s, pi(s)) in reward units, one entry per RL row."""
    from src.rl.analysis import data as D
    from src.rl.evaluate_cql_alpha_sweep import actor_actions
    s, a = D.states(processor), D.actions(processor)
    a_pi = actor_actions(cql, s, batch_size=batch_size)
    scale = float(fqe.reward_scale)
    return (np.asarray(fqe.q(s, a, batch_size=batch_size, scaled=True), dtype=np.float64) * scale,
            np.asarray(fqe.q(s, a_pi, batch_size=batch_size, scaled=True), dtype=np.float64) * scale)


def doubly_robust_one_seed(cql, bc, fqe, processor, split, wcap_values, gamma, cache_dir=None, tag=None,
                           n_boot=200, boot_seed=0, sarsa=None, lp_kwargs=None, verbose=True):
    """DR rows (one per weight cap) for one matched (CQL, BC, FQE) triple on `processor`."""
    from src.rl import evaluate_wis as W
    from src.rl.evaluate_fqe_final_seeds import check_fqe_matches
    from src.rl.evaluate_cql_alpha_sweep import check_judge_matches

    check_judge_matches(cql, bc)
    check_fqe_matches(cql, fqe, sarsa)
    b_lp, t_lp = W.get_wis_log_probs(cql, bc, processor, split, cache_dir=cache_dir, tag=tag, **(lp_kwargs or {}))
    trajs, _ = W.build_trajectory_data(processor, b_lp, t_lp)
    q_row, v_row = _critic_values(cql, fqe, processor)

    LR, R, Q, V, L = dense_matrices([t["log_ratio"] for t in trajs], [t["rewards"] for t in trajs],
                                    [q_row[t["start"]:t["end"]] for t in trajs],
                                    [v_row[t["start"]:t["end"]] for t in trajs], [t["length"] for t in trajs])
    n = len(L)
    rng = np.random.default_rng(boot_seed)
    boots = rng.integers(0, n, size=(n_boot, n))

    G0 = float(W.discounted_returns(trajs, gamma).mean())
    if sarsa is not None:                                            # unit check: SARSA clinician value vs observed return
        from src.rl.analysis import data as D
        first = np.array([t["start"] for t in trajs])
        v_mu = np.asarray(sarsa.q(D.states(processor), D.actions(processor), scaled=True))[first] * float(sarsa.reward_scale)
        if verbose:
            print(f"  unit check: SARSA V_mu {v_mu.mean():+.3f} vs observed return G0 {G0:+.3f} (reward units)")

    rows = []
    for cap in wcap_values:
        est = dr_estimate(LR, R, Q, V, L, gamma, cap)
        ref = W.compute_stepwise_wis(trajs, gamma, weight_cap_percentile=cap)["value"]
        ref_null = W.compute_stepwise_wis(W._null_trajectories(trajs), gamma, weight_cap_percentile=cap)["value"]
        for name, mine, theirs in (("stepwise WIS", est["step"], ref), ("stepwise null", est["null"], ref_null),
                                   ("DR null", est["dr_null"], ref_null)):
            if abs(mine - theirs) > 1e-6:
                raise AssertionError(f"seed {cql.seed} cap {cap}: {name} {mine:.6f} does not match evaluate_wis {theirs:.6f}")
        bs = [dr_estimate(LR, R, Q, V, L, gamma, cap, idx=ix) for ix in boots]
        d_dr = np.array([b["dr"] - b["null"] for b in bs])
        d_is = np.array([b["step"] - b["null"] for b in bs])
        row = {"seed": cql.seed, "cap": cap, "n_stays": n,
               "dr": est["dr"], "null": est["null"], "dr_minus_null": est["dr"] - est["null"],
               "dr_lo": float(np.percentile(d_dr, 2.5)), "dr_hi": float(np.percentile(d_dr, 97.5)),
               "is_minus_null": est["step"] - est["null"],
               "is_lo": float(np.percentile(d_is, 2.5)), "is_hi": float(np.percentile(d_is, 97.5)),
               "dm": est["dm"], "dm_minus_G0": est["dm"] - G0, "G0": G0}
        rows.append(row)
        if verbose:
            print(f"  cap {cap:>5g}: DR - null {row['dr_minus_null']:+.3f} [{row['dr_lo']:+.3f}, {row['dr_hi']:+.3f}]   "
                  f"IS - null {row['is_minus_null']:+.3f} [{row['is_lo']:+.3f}, {row['is_hi']:+.3f}]   "
                  f"critic alone V_pi - G0 {row['dm_minus_G0']:+.3f}")
    return rows


def run_doubly_robust(cql_models, bc_models, fqe_models, store, split="val", wcap_values=(10, 25, 50), gamma=None,
                      cache_dir=None, tag="auto", n_boot=200, sarsa_models=None, boot_seed=0, verbose=True, **lp_kwargs):
    """
    DR for every seed: each CQL seed with its own BC seed, FQE critic and state embedding.
    wcap_values : weight-cap percentiles (same meaning as in the WIS sweep); ESS >= ~60% means cap <= 50.
    gamma=None  : the CQL models' saved discount. tag/cache_dir: as in run_wis_for_models (reuses the cached log-probs,
                  including the coarsened ones when wis.ACTION_COARSENING is set).
    sarsa_models: optional; prints a unit check of SARSA V_mu against the observed return.
    Returns a DataFrame with one row per (seed, cap); see summarize_doubly_robust.
    """
    from src.rl import evaluate_wis as W
    from src.rl.evaluate_fqe_final_seeds import _by_seed
    cql, bc, fqe = _by_seed(cql_models), _by_seed(bc_models), _by_seed(fqe_models)
    sarsa = _by_seed(sarsa_models) if sarsa_models is not None else {}
    missing = sorted((set(cql) - set(bc)) | (set(cql) - set(fqe)), key=str)
    if missing:
        raise KeyError(f"seeds {missing} lack a BC or FQE model")
    gamma = W._resolve_gamma(gamma, cql_models)
    if cache_dir and tag == "auto":
        tag = W.cql_alpha_tag(cql_models)
    elif tag == "auto":
        tag = None
    if verbose:
        print(f"action coarsening: {W.ACTION_COARSENING}")
    rows = []
    for seed in sorted(cql, key=str):
        if verbose:
            print(f"\n=== Seed {seed} ===")
        rows += doubly_robust_one_seed(cql[seed], bc[seed], fqe[seed], store.processor(cql[seed], split), split,
                                       wcap_values, gamma, cache_dir=cache_dir, tag=tag, n_boot=n_boot,
                                       boot_seed=boot_seed, sarsa=sarsa.get(seed), lp_kwargs=lp_kwargs, verbose=verbose)
    return pd.DataFrame(rows)


# =============================================================================
# Summary and plot
# =============================================================================

def summarize_doubly_robust(df):
    """Across-seed mean / min / max of DR - null and IS - null per cap, the number of seeds whose CI excludes 0, and the critic-alone value."""
    g = df.groupby("cap")
    out = pd.DataFrame({
        "dr_mean": g.dr_minus_null.mean(), "dr_min": g.dr_minus_null.min(), "dr_max": g.dr_minus_null.max(),
        "dr_seeds_ci_above_0": df.assign(_p=df.dr_lo > 0).groupby("cap")["_p"].sum().astype(int),
        "is_mean": g.is_minus_null.mean(), "is_min": g.is_minus_null.min(), "is_max": g.is_minus_null.max(),
        "critic_alone_mean": g.dm_minus_G0.mean(), "n_seeds": g.seed.nunique()})
    return out


def plot_doubly_robust(df, title="", save_path=None):
    """Per seed and cap: DR - null with bootstrap CI (left) and importance weights alone, IS - null (right). Same y-scale."""
    import matplotlib.pyplot as plt
    caps = sorted(df.cap.unique())
    seeds = sorted(df.seed.unique(), key=str)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, (col, lo, hi, name) in zip(axes, (("dr_minus_null", "dr_lo", "dr_hi", "doubly robust"),
                                              ("is_minus_null", "is_lo", "is_hi", "importance weights only"))):
        ax.axhline(0, color="grey", lw=1)
        for j, seed in enumerate(seeds):
            d = df[df.seed == seed].set_index("cap").loc[caps]
            x = np.arange(len(caps)) + (j - (len(seeds) - 1) / 2) * 0.12
            ax.errorbar(x, d[col], yerr=[d[col] - d[lo], d[hi] - d[col]], fmt="o", ms=3, capsize=2, lw=1, label=f"seed {seed}")
        ax.set_xticks(np.arange(len(caps)), [f"{c:g}" for c in caps])
        ax.set_xlabel("weight cap (percentile)"); ax.set_title(name)
    axes[0].set_ylabel("policy - clinician (reward units, stepwise null removed)")
    axes[0].legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig