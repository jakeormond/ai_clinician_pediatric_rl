"""
src/rl/evaluate_fqe_final_seeds.py

Off-policy evaluation of the FINAL CQL models (one per seed) with fitted Q evaluation (FQE).

Loading goes through runs.py / models.py / data.py (LoadedModel, DataStore); this file holds
only the OPE-specific code. Notebook: evaluate_fqe_final_seeds.ipynb.

What FQE can and cannot say
  An FQE critic for seed k was fit to the value of ONE target policy: CQL seed k's actor,
  plugged into its own Bellman backup. So Q^pi_k(s, a) = "take a now, then follow CQL seed k".
  It is valid for CQL seed k and for no other seed. The pairing is enforced here:
    CQL seed k  <->  FQE seed k  <->  SARSA seed k  (and all three on Dynamic-DeepHit seed k).

Two different quantities are reported (they are NOT the same thing):

  adv1  = mean Q^pi(s, pi(s)) - Q^pi(s, a_clinician)
          One-step advantage under pi's continuation. Uses the FQE critic only. This is what
          the earlier version of this file called "advantage". It says whether, GIVEN that
          pi acts from the next step on, pi's action now beats the clinician's action now.
          It is not the difference between the two policies' values.

  dV    = V_pi - V_mu, with
          V_pi = mean over INITIAL states of Q^pi_FQE(s0, pi(s0))     (the FQE critic)
          V_mu = mean over INITIAL states of Q^mu_SARSA(s0, a0)       (seed-matched SARSA:
                 value of the clinician policy, continuing as logged)
          This is the policy-value comparison. It needs `sarsa_models`. The two numbers come
          from two different critics fit to the same data by the same TD method, so shared
          biases partly cancel, but it is still an estimate with model error: report it with
          the support (occupancy) of pi's actions next to it.

Units: health-score points (Q / total_health_score_scale of each critic's own run).

Uncertainty
  *_lo / *_hi are 95% percentile intervals from a cluster bootstrap over patient stays
  (trajectories), paired across quantities. They reflect sampling of patients for ONE seed,
  not training variability; that is the spread across seeds (see summarize_fqe).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rl.analysis import data as D
from src.rl.evaluate_cql_alpha_sweep import (actor_actions, check_judge_matches, build_occupancy_table,
                                             occupancy_support_of_actions)


# =============================================================================
# Pairing
# =============================================================================

def _by_seed(models) -> Dict[Any, Any]:
    """{seed: model} from {seed: model} or from a Study result with exactly one group."""
    if models and all(isinstance(k, tuple) for k in models):
        if len(models) != 1:
            raise ValueError(f"expected one group of models, found groups {list(models)}; pass one group, "
                             f"e.g. cql_models[(0.029,)]")
        return dict(next(iter(models.values())))
    return dict(models)


_POLICY_KEY_HINTS = ("target_policy", "policy_path", "actor_path", "cql")


def target_policy_hints(fqe) -> Dict[str, str]:
    """String-valued saved hyperparameters of an FQE run that look like they point at its target policy."""
    return {k: v for k, v in fqe.record.hyperparams.items()
            if isinstance(v, str) and any(t in k.lower() for t in _POLICY_KEY_HINTS)}


def _same_or_under(path, ref) -> bool:
    a, b = os.path.normpath(str(path)), os.path.normpath(str(ref))
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def _reward_description(model) -> Dict[str, Any]:
    return {"discount": model.record.get("discount"), "reward": dict(D.reward_signature(model.record))}


def check_fqe_matches(cql, fqe, sarsa=None, target_policy_check="warn", reward_check="raise"):
    """
    Raise unless the FQE (and SARSA, if given) critic has the same seed and state embedding as
    the CQL model. Also looks for the FQE run's recorded target policy and compares it with the
    CQL run folder: target_policy_check = 'warn' (default) | 'raise' | 'off'.

    With SARSA, dV subtracts a SARSA value from an FQE value, which only means something if both
    critics were fit with the same discount and reward settings: reward_check = 'raise' (default)
    | 'warn' | 'off' compares them.
    Returns a dict describing the pairing, for the printed pairing table.
    """
    check_judge_matches(cql, fqe)
    if sarsa is not None:
        check_judge_matches(cql, sarsa)
        a, b = _reward_description(fqe), _reward_description(sarsa)
        if a != b and reward_check != "off":
            msg = (f"seed {cql.seed}: FQE and SARSA were fit with different discount/reward settings, so V_pi - V_mu "
                   f"mixes units.\n  FQE  : {a}\n  SARSA: {b}")
            if reward_check == "raise":
                raise ValueError(msg + "\n(pass reward_check='warn' to continue anyway)")
            warnings.warn(msg)

    hints = target_policy_hints(fqe)
    status = "no target-policy path recorded in FQE hyperparameters"
    if hints:
        ok = any(_same_or_under(cql.record.run_dir, v) for v in hints.values())
        status = "recorded target policy matches this CQL run" if ok else "recorded target policy DOES NOT match this CQL run"
        if not ok and target_policy_check != "off":
            msg = (f"{fqe.record.label}: FQE records target policy {hints}, which does not contain / equal the CQL run "
                   f"{cql.record.run_dir}. An FQE critic is only valid for the policy it was trained against.")
            if target_policy_check == "raise":
                raise ValueError(msg)
            warnings.warn(msg)
    elif target_policy_check != "off":
        warnings.warn(f"{fqe.record.label}: cannot verify which CQL model this FQE was trained against "
                      f"(no target-policy path in its hyperparameters). Confirm from the training log.")
    return {"seed": cql.seed, "cql_run": cql.record.run_dir, "fqe_run": fqe.record.run_dir,
            "fqe_target_hint": hints or None, "check": status}


# =============================================================================
# Per-seed OPE
# =============================================================================

def first_row_mask(traj: np.ndarray) -> np.ndarray:
    """True on the first RL row of each trajectory (the initial state of a stay)."""
    return np.concatenate(([True], traj[1:] != traj[:-1]))


def _pct_ci(x: np.ndarray, level=0.95):
    a = (1 - level) / 2
    lo, hi = np.quantile(x, [a, 1 - a])
    return float(lo), float(hi)


def fqe_ope_one_seed(cql, fqe, processor, sarsa=None, occupancy_table=None, min_count=10,
                     n_boot=1000, boot_seed=0, batch_size=4096) -> Dict[str, Any]:
    """OPE quantities for one matched (CQL, FQE[, SARSA]) triple on `processor` (see module docstring)."""
    s = D.states(processor)
    a_clin = D.actions(processor)
    a_pi = actor_actions(cql, s, batch_size=batch_size)           # deterministic, as everywhere else

    q_pi_pi = fqe.q(s, a_pi, batch_size=batch_size, scaled=True)       # Q^pi(s, pi(s))
    q_pi_cl = fqe.q(s, a_clin, batch_size=batch_size, scaled=True)     # Q^pi(s, a_clin)
    adv_all = q_pi_pi - q_pi_cl

    traj = D.trajectory_ids(processor)
    first = first_row_mask(traj)
    inv = np.unique(traj, return_inverse=True)[1]
    J = int(inv.max()) + 1
    if int(first.sum()) != J:
        raise ValueError(f"{cql.record.label}: {first.sum()} initial rows but {J} trajectories; rows are not contiguous by stay")

    v_pi0, q_cl0 = q_pi_pi[first], q_pi_cl[first]
    adv0 = v_pi0 - q_cl0
    S = np.bincount(inv, weights=adv_all, minlength=J)
    n = np.bincount(inv, minlength=J).astype(np.float64)

    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, J, size=(n_boot, J), dtype=np.int32)         # shared resamples: paired CIs

    row: Dict[str, Any] = {"seed": cql.seed, "alpha": cql.record.get("cql_alpha"),
                           "n_rows": len(s), "n_stays": J}

    row["V_pi"] = float(v_pi0.mean())
    row["V_pi_lo"], row["V_pi_hi"] = _pct_ci(v_pi0[idx].mean(1))

    row["adv1"] = float(adv0.mean())
    row["adv1_lo"], row["adv1_hi"] = _pct_ci(adv0[idx].mean(1))

    iqr = max(float(np.quantile(q_pi_cl, 0.75) - np.quantile(q_pi_cl, 0.25)), 1e-6)
    row["adv_all"] = float(adv_all.mean())
    row["adv_all_lo"], row["adv_all_hi"] = _pct_ci(S[idx].sum(1) / n[idx].sum(1))
    row["adv_all_norm"] = row["adv_all"] / iqr
    row["frac_adv_pos"] = float((adv_all > 0).mean())
    row["fqe_q_iqr"] = iqr

    if sarsa is not None:
        v_mu0 = sarsa.q(s, a_clin, batch_size=batch_size, scaled=True)[first]
        d0 = v_pi0 - v_mu0
        row["V_mu"] = float(v_mu0.mean())
        row["V_mu_lo"], row["V_mu_hi"] = _pct_ci(v_mu0[idx].mean(1))
        row["dV"] = float(d0.mean())
        row["dV_lo"], row["dV_hi"] = _pct_ci(d0[idx].mean(1))

    if occupancy_table is not None:
        hs, _ = D.health_score(processor)
        row["occupancy"], _ = occupancy_support_of_actions(a_pi, hs, occupancy_table, min_count=min_count)
        row["occupancy"] = float(row["occupancy"])
    return row


def run_fqe_ope(cql_models, fqe_models, store, sarsa_models=None, split="val", n_boot=1000,
                min_count=10, n_severity_bins=20, n_action_bins=10, boot_seed=0,
                target_policy_check="warn", reward_check="raise", verbose=True) -> pd.DataFrame:
    """
    cql_models  : the FINAL CQL models, {seed: LoadedModel} or one group e.g. cql_hs_models[(0.029,)]
    fqe_models  : load_study_models(fqe_study), {(): {seed: LoadedModel}}
    sarsa_models: optional, seed-matched SARSA critics (enables V_mu and dV)
    store       : data.DataStore. Use keep=("train", "val", "test"): train is needed for the occupancy
                  table, and test for the confirmation run.
    split       : 'val' for development, 'test' once, as confirmation.

    Returns one row per seed (see module docstring for the columns).
    """
    cql, fqe = _by_seed(cql_models), _by_seed(fqe_models)
    sarsa = _by_seed(sarsa_models) if sarsa_models is not None else None

    missing = sorted((set(cql) - set(fqe)) | (set(cql) - (set(sarsa) if sarsa else set(cql))), key=str)
    if missing:
        raise KeyError(f"no FQE{'/SARSA' if sarsa else ''} critic for CQL seeds {missing}; "
                       f"FQE seeds {sorted(fqe, key=str)}" + (f", SARSA seeds {sorted(sarsa, key=str)}" if sarsa else ""))

    pairing, rows = [], []
    for seed in sorted(cql, key=str):
        m, f = cql[seed], fqe[seed]
        sa = sarsa[seed] if sarsa else None
        pairing.append(check_fqe_matches(m, f, sa, target_policy_check, reward_check))
        p = store.processor(m, split)
        table = build_occupancy_table(store.processor(m, "train"), n_severity_bins, n_action_bins)
        rows.append(fqe_ope_one_seed(m, f, p, sarsa=sa, occupancy_table=table, min_count=min_count,
                                     n_boot=n_boot, boot_seed=boot_seed))
        if verbose:
            r = rows[-1]
            msg = f"  seed {seed}: V_pi {r['V_pi']:.2f} [{r['V_pi_lo']:.2f}, {r['V_pi_hi']:.2f}]  adv1 {r['adv1']:.3f}"
            if "dV" in r:
                msg += f"  V_mu {r['V_mu']:.2f}  dV {r['dV']:.3f} [{r['dV_lo']:.3f}, {r['dV_hi']:.3f}]"
            print(msg + f"  occupancy {r['occupancy']:.3f}")

    if verbose:
        print("\nPairing (CQL seed <-> FQE seed):")
        print(pd.DataFrame(pairing).to_string(index=False))
    return pd.DataFrame(rows)


# =============================================================================
# Summary and plot
# =============================================================================

def summarize_fqe(df: pd.DataFrame) -> pd.DataFrame:
    """Across-seed summary (mean / min / max / sd) of the main quantities, plus how many seeds have CI > 0."""
    cols = [c for c in ("V_pi", "V_mu", "dV", "adv1", "adv_all_norm", "frac_adv_pos", "occupancy") if c in df]
    out = df[cols].agg(["mean", "min", "max", "std"]).T
    pos = {c: int((df[f"{c}_lo"] > 0).sum()) for c in ("dV", "adv1") if f"{c}_lo" in df}
    out["seeds_with_CI_above_0"] = pd.Series(pos)
    return out


def plot_fqe_ope(df: pd.DataFrame, title: str = "", save_path: Optional[str] = None):
    """Per-seed point estimates with bootstrap CIs: policy values (left) and the two advantages (right)."""
    has_mu = "dV" in df
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(df))

    ax = axes[0]
    ax.errorbar(x - 0.1, df.V_pi, yerr=[df.V_pi - df.V_pi_lo, df.V_pi_hi - df.V_pi], fmt="o", capsize=3, label="CQL policy (FQE)")
    if has_mu:
        ax.errorbar(x + 0.1, df.V_mu, yerr=[df.V_mu - df.V_mu_lo, df.V_mu_hi - df.V_mu], fmt="s", capsize=3, label="clinician (SARSA)")
    ax.set_xticks(x, df.seed.astype(str)); ax.set_xlabel("seed"); ax.set_ylabel("value at initial states (health-score points)")
    ax.legend(frameon=False); ax.set_title("policy value")

    ax = axes[1]
    ax.axhline(0, color="grey", lw=1)
    if has_mu:
        ax.errorbar(x - 0.1, df.dV, yerr=[df.dV - df.dV_lo, df.dV_hi - df.dV], fmt="o", capsize=3, label="dV = V_pi - V_mu")
    ax.errorbar(x + 0.1, df.adv1, yerr=[df.adv1 - df.adv1_lo, df.adv1_hi - df.adv1], fmt="s", capsize=3, label="adv1 (one-step, FQE only)")
    ax.set_xticks(x, df.seed.astype(str)); ax.set_xlabel("seed"); ax.set_ylabel("health-score points")
    ax.legend(frameon=False); ax.set_title("difference from clinician")
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig