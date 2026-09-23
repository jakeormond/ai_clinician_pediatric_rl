"""
Weighted importance sampling (WIS) off-policy evaluation of the CQL policy,
using the BC actor's `log_prob_of_action` as the behavior-policy density
pi_b(a|s) and the CQL actor's `log_prob_for_wis` (which borrows BC's fitted
per-state std as bandwidth around the CQL policy's deterministic action) as
the evaluation density pi_e(a|s).

Seed pairing follows the SAME discipline established for CQL<->FQE pairing
in evaluate_fqe_final_seeds.py: each CQL seed's behavior density MUST come
from its matched BC seed (not an arbitrary one), and the seed itself is read
from each model's own results.json['hyperparameters']['seed'] -- not from a
config_manager.pkl, which this project has already found to sometimes be
shared across an entire seed sweep rather than saved per-run.

Two WIS estimators are provided, per project convention of reporting more
than one angle on the same question (raw vs. normalized advantage, own- vs.
SARSA-critic, etc.):

  1. Trajectory-level (whole-episode) WIS -- one cumulative importance ratio
     per episode, applied to that episode's total discounted return.
  2. Stepwise / per-decision WIS (Precup et al. / AI-Clinician convention)
     -- the importance weight is accumulated timestep-by-timestep and
     combined across trajectories AT each timestep, before discounting and
     summing over time. Lower variance than (1) for long horizons, and the
     standard choice in this literature.

Both share the same underlying per-timestep log importance ratios (with
optional percentile capping applied once, upstream, so the two estimators
are always compared on the same stabilized weights) and the same
bootstrap-over-trajectories machinery for confidence intervals.

Analysis-workflow note (see project_summary_claude_19September2026.md):
this file holds the functions; the paired notebook (evaluate_wis.ipynb)
runs the actual analysis, calls, and plots.
"""

import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats


def _to_numpy(x):
    """
    Coerces a torch.Tensor to a detached CPU numpy array; leaves numpy
    arrays (or anything else) unchanged.

    Needed because processor.rl_dataset's fields (rewards, dones,
    mortality, ...) may be stored as torch tensors rather than numpy
    arrays -- mixing a torch tensor into numpy math (e.g. t["rewards"] *
    a numpy array, then np.sum(...) on the result) silently returns a
    torch Tensor instead of raising immediately, and then fails later
    with a confusing TypeError about axis/out kwargs when numpy's
    reduction functions hit torch's incompatible method signature. Call
    this once, right where each field is pulled out of the processor, so
    everything downstream is unambiguously numpy.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _to_numpy_1d(x):
    """
    _to_numpy, plus: squeezes a trailing singleton dimension, e.g. shape
    (N, 1) -> (N,). Scalar-per-timestep fields (rewards, dones, mortality)
    are sometimes stored with a trailing dim of 1, matching the (batch, 1)
    convention used elsewhere in this project (e.g. TrainerUtils'
    unpacked reward/done tensors, critic outputs). Left un-squeezed, that
    shape silently broadcasts wrong in later numpy math (e.g. (L,1) * (L,)
    -> (L,L)) instead of raising -- this catches that case explicitly by
    raising instead of guessing, if the field isn't actually scalar-per-row.
    """
    x = _to_numpy(x)
    if x.ndim > 1:
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != 1:
            raise ValueError(
                f"Expected a scalar-per-timestep field (shape (N,) or (N,1)), got shape {x.shape}"
            )
        x = flat[:, 0]
    return x


# ---------------------------------------------------------------------------
# Color palette -- consistent with the rest of the project's analysis plots
# (see e.g. evaluate_cql_alpha_sweep.py's OWN_COLOR/SARSA_COLOR/OCC_COLOR).
# ---------------------------------------------------------------------------
STEPWISE_COLOR = "#2471a3"      # blue -- primary/recommended estimator
TRAJECTORY_COLOR = "#7d3c98"    # purple -- secondary estimator
BASELINE_COLOR = "#7f8c8d"      # grey -- observed clinician baseline
ESS_COLOR = "#229954"           # green -- effective-sample-size / support diagnostics
WEIGHT_COLOR = "#c0392b"        # red -- weight-concentration diagnostics


# ---------------------------------------------------------------------------
# Seed matching (CQL <-> BC), mirroring match_cql_fqe_pairs in
# evaluate_fqe_final_seeds.py, but reading the seed from results.json rather
# than a training checkpoint (per your convention: results.json holds
# hyperparameters['seed'] for both CQL and BC runs).
# ---------------------------------------------------------------------------

def get_seed_from_results_json(model_dir):
    """
    Reads the training seed from a model's OWN results.json
    (results['hyperparameters']['seed']). Mirrors
    evaluate_fqe_final_seeds.get_seed_from_checkpoint's rationale: read the
    run-specific source of truth, never a possibly-shared config_manager.pkl.
    """
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    if "hyperparameters" not in results or "seed" not in results["hyperparameters"]:
        raise KeyError(
            f"No hyperparameters['seed'] found in {results_path}. "
            f"Available hyperparameter keys: {list(results.get('hyperparameters', {}).keys())}"
        )
    return results["hyperparameters"]["seed"]


def match_cql_bc_pairs(cql_models_with_seeds, bc_models_with_seeds):
    """
    Matches each CQL model to the BC model that should supply its
    behavior-policy density, using each model's OWN seed -- read directly
    from the hyperparams dict already returned by the loader
    (hyperparams['seed']), never re-derived from, or assumed aligned
    with, a separate directory list.

    cql_models_with_seeds: list of (actor, critic, config_manager,
        hyperparams) tuples, as returned by etm.load_actor_and_critic
        for model_type='cql'.
    bc_models_with_seeds: list of (actor, config_manager, hyperparams)
        tuples, as returned by etm.load_actor_and_critic for
        model_type='bc'.

    Matching works by sorting each list independently by its own seed,
    then zipping the two sorted lists position-wise -- valid precisely
    because sorting by the same key produces the same relative order on
    both sides, so position i after sorting is "the model with the i-th
    smallest seed" on each side, and (with matching seed sets) that's the
    same seed on both sides at every position.

    NOTE (flagging for your own use, not changed here since you've
    verified this against your real 5-seed models): this version does
    NOT raise if the two sides have duplicate seeds, or if their seed
    sets don't match exactly -- the seed-mismatch guard in the pairs
    list comprehension just silently drops that one position's pair
    instead of erroring, so e.g. a missing seed on one side quietly
    returns 4 pairs instead of 5 rather than telling you why. Given this
    project's history with silent seed-mismatches (config_manager.pkl),
    it's worth eyeballing that len(pairs) == 5 (or your expected seed
    count) after calling this, especially if you ever load a different
    set of CQL/BC models than the ones you've already checked.

    Returns (pairs, cql_models_with_seeds, bc_models_with_seeds), where
    pairs is a list of (cql_actor, bc_actor, seed) tuples sorted by seed,
    and the two model lists are returned sorted by seed as well (they are
    also sorted IN PLACE as a side effect, since list.sort() mutates).
    """
    cql_seeds = [hyperparams['seed'] for _, _, _, hyperparams in cql_models_with_seeds]
    bc_seeds = [hyperparams['seed'] for _, _, hyperparams in bc_models_with_seeds]

    print(f"CQL seeds loaded: {sorted(cql_seeds)}")
    print(f"BC seeds loaded:  {sorted(bc_seeds)}")

    # reorder the lists by seed, so the returned pairs are always in a consistent order
    cql_models_with_seeds.sort(key=lambda x: x[3]['seed'])
    bc_models_with_seeds.sort(key=lambda x: x[2]['seed'])

    pairs = [(cql_model, bc_model, seed) for (cql_model, _, _, cql_hyperparams), (bc_model, _, bc_hyperparams) in zip(cql_models_with_seeds, bc_models_with_seeds) for seed in [cql_hyperparams['seed']] if seed == bc_hyperparams['seed']]
    print(f"Matched {len(pairs)} (CQL, BC) pairs by results.json seed: {[p[2] for p in pairs]}")

    return pairs, cql_models_with_seeds, bc_models_with_seeds


def get_cql_actor(cql_model_entry):
    """CQL tuples are (actor, critic, config_manager, hyperparams)."""
    return cql_model_entry[0] if isinstance(cql_model_entry, (list, tuple)) else cql_model_entry


def get_bc_actor(bc_model_entry):
    """
    BC tuples are assumed (actor, config_manager, hyperparams) -- no
    critic, analogous to how SARSA/FQE tuples have no separate actor.
    If your BC loader returns a different shape, adjust this accessor only;
    everything downstream just calls the returned object as an nn.Module
    with .log_prob_of_action / .eval() / .forward(..., return_dist=True).
    """
    return bc_model_entry[0] if isinstance(bc_model_entry, (list, tuple)) else bc_model_entry


# ---------------------------------------------------------------------------
# Trajectory boundaries -- identical logic to
# evaluate_cql_alpha_sweep.get_actor_actions's trajectory loop, factored out
# so it isn't duplicated across analysis files.
# ---------------------------------------------------------------------------

def get_trajectory_boundaries(processor):
    """
    Returns a list of (start, end, stay_id) tuples giving each trajectory's
    half-open [start, end) row range into processor.rl_dataset's flat
    arrays, in the same order as processor.dynamic_deephit_dataset['stay_ids'].
    """
    dones = _to_numpy_1d(processor.rl_dataset["dones"])
    traj_end_inds = np.where(dones == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = _to_numpy_1d(processor.dynamic_deephit_dataset["stay_ids"])

    boundaries = []
    for stay_idx, stay_id in enumerate(stay_ids):
        # positional lookup matches get_actor_actions' np.where(...)[0][0]
        # under the assumption (true elsewhere in this project) that
        # stay_ids is unique and ordered the same as the trajectories in
        # rl_dataset. Falls back to an explicit search if that assumption
        # is ever violated, rather than silently mis-slicing a trajectory.
        if stay_idx + 1 >= len(traj_start_inds):
            break
        start, end = traj_start_inds[stay_idx], traj_start_inds[stay_idx + 1]
        boundaries.append((int(start), int(end), stay_id))

    if len(boundaries) != len(stay_ids):
        raise ValueError(
            f"Recovered {len(boundaries)} trajectories but stay_ids has "
            f"{len(stay_ids)} entries -- dones-based segmentation may not "
            f"line up 1:1 with stay_ids for this processor. Check for "
            f"trailing/leading mismatches before trusting downstream results."
        )
    return boundaries


# ---------------------------------------------------------------------------
# Per-timestep log-probabilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_behavior_and_target_log_probs(
        cql_actor, bc_actor, states, actions,
        batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    For every (state, action) row in the dataset, computes:
      behavior_lp[i] = log pi_b(action_i | state_i)   via bc_actor.log_prob_of_action
      target_lp[i]   = log pi_e(action_i | state_i)   via cql_actor.log_prob_for_wis(..., bc_actor)

    log_prob_for_wis constructs its evaluation density around the CQL
    actor's DETERMINISTIC action using BC's per-state std as bandwidth
    (see cql_with_logprobs.Actor.log_prob_for_wis's docstring) -- this is
    the density this project has already built specifically for WIS, so we
    call it directly rather than re-deriving anything here.

    Returns two float64 numpy arrays of shape (n,).
    """
    cql_actor.eval()
    bc_actor.eval()

    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.float32)
    n = states_t.shape[0]

    behavior_lp = np.zeros(n, dtype=np.float64)
    target_lp = np.zeros(n, dtype=np.float64)

    for i in range(0, n, batch_size):
        s = states_t[i:i + batch_size].to(device)
        a = actions_t[i:i + batch_size].to(device)

        b_lp = bc_actor.log_prob_of_action(s, a)
        t_lp = cql_actor.log_prob_for_wis(s, a, bc_actor)

        behavior_lp[i:i + batch_size] = b_lp.detach().cpu().numpy()
        target_lp[i:i + batch_size] = t_lp.detach().cpu().numpy()

    return behavior_lp, target_lp


# ---------------------------------------------------------------------------
# Weight capping (optional stabilization)
# ---------------------------------------------------------------------------

def compute_log_ratio_cap_bounds(log_ratio_flat, cap_percentile):
    """
    Computes symmetric [cap_percentile, 100 - cap_percentile] bounds from
    the POOLED per-timestep log importance ratios across the whole
    dataset (all trajectories, before splitting) -- i.e. capping the raw
    per-decision log-ratio, before any cumulative summing, so a single
    clinical decision can't dominate either estimator.

    Directly motivated by BC's own docstring finding that the fluid gate's
    log-likelihood can account for a median 33-39% (p90 73-82%) of a whole
    trajectory's gate-related log-likelihood magnitude -- exactly the kind
    of single-timestep dominance WIS weight-capping guards against.
    """
    lower = np.percentile(log_ratio_flat, cap_percentile)
    upper = np.percentile(log_ratio_flat, 100 - cap_percentile)
    return lower, upper


def build_trajectory_data(processor, behavior_lp, target_lp, cap_percentile=None):
    """
    Splits flat per-timestep behavior/target log-probs, rewards, and
    mortality into a list of per-trajectory dicts using
    get_trajectory_boundaries. If cap_percentile is given (e.g. 1.0 for a
    1st/99th-percentile cap), the per-timestep log-ratio is clipped ONCE,
    here, before splitting -- so trajectory-level and stepwise WIS always
    see the same stabilized weights and are directly comparable.

    Each trajectory dict has:
      stay_id, start, end (this trajectory's row range into the flat
      processor arrays -- kept for diagnostics, e.g. find_dominant_trajectories),
      log_ratio (length-L array, capped if cap_percentile was given),
      behavior_lp, target_lp (length-L arrays, always RAW/uncapped, so
      diagnostics can show the true magnitude of a mismatch even when
      log_ratio itself has been capped for the estimators), rewards
      (length-L array), mortality (scalar, taken from this trajectory's
      final timestep), length (L)

    Returns (trajectories, cap_bounds) where cap_bounds is (lower, upper)
    or (None, None) if cap_percentile was not given.
    """
    rewards = _to_numpy_1d(processor.rl_dataset["rewards"])
    mortality = _to_numpy_1d(processor.rl_dataset["mortality"])

    log_ratio_flat = target_lp - behavior_lp
    cap_bounds = (None, None)
    if cap_percentile is not None:
        lower, upper = compute_log_ratio_cap_bounds(log_ratio_flat, cap_percentile)
        n_clipped = int(np.sum((log_ratio_flat < lower) | (log_ratio_flat > upper)))
        print(f"  Capping log-ratios to [{lower:.3f}, {upper:.3f}] "
              f"({cap_percentile:.1f}/{100 - cap_percentile:.1f} pctile): "
              f"{n_clipped} / {len(log_ratio_flat)} timesteps affected "
              f"({100 * n_clipped / len(log_ratio_flat):.2f}%)")
        log_ratio_flat = np.clip(log_ratio_flat, lower, upper)
        cap_bounds = (lower, upper)

    boundaries = get_trajectory_boundaries(processor)
    trajectories = []
    for start, end, stay_id in boundaries:
        if end <= start:
            continue
        trajectories.append({
            "stay_id": stay_id,
            "start": start,
            "end": end,
            "log_ratio": log_ratio_flat[start:end],
            "behavior_lp": behavior_lp[start:end],
            "target_lp": target_lp[start:end],
            "rewards": rewards[start:end],
            "mortality": mortality[start:end][-1],
            "length": end - start,
        })
    return trajectories, cap_bounds


# ---------------------------------------------------------------------------
# WIS estimators
# ---------------------------------------------------------------------------

def compute_ess(weights):
    """Kish's effective sample size: (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=np.float64)
    denom = np.sum(w ** 2)
    if denom == 0:
        return 0.0
    return (w.sum() ** 2) / denom


def discounted_returns(trajectories, gamma):
    """Per-trajectory discounted return under the REAL (logged) rewards."""
    return np.array([
        np.sum(t["rewards"] * (gamma ** np.arange(len(t["rewards"]))))
        for t in trajectories
    ])


def compute_trajectory_wis(trajectories, gamma):
    """
    Standard (whole-episode) weighted importance sampling:

        V_WIS = sum_i w_i * G_i / sum_i w_i,   w_i = exp(sum_t log_ratio_t^i)

    WIS reweights which TRAJECTORIES count more; it never alters the
    observed reward sequence. log-weights are normalized by their max
    before exponentiating (numerically stable, does not change the ratio).
    """
    log_weights = np.array([np.sum(t["log_ratio"]) for t in trajectories])
    returns = discounted_returns(trajectories, gamma)

    max_lw = log_weights.max()
    weights = np.exp(log_weights - max_lw)
    wis_value = np.sum(weights * returns) / np.sum(weights)

    return {
        "value": wis_value,
        "weights": weights,
        "log_weights": log_weights,
        "returns": returns,
        "ess": compute_ess(weights),
        "n_trajectories": len(trajectories),
    }


def compute_stepwise_wis(trajectories, gamma):
    """
    Per-decision (stepwise) WIS, following the Precup et al. / AI-Clinician
    convention: the importance weight is accumulated timestep-by-timestep
    and combined across trajectories AT each timestep (rather than once
    per whole episode), then discounted and summed over time:

        w_t^i = exp( sum_{t'=0}^{t} log_ratio_{t'}^i )
        V_stepwise = sum_t gamma^t * [ sum_i w_t^i r_t^i ] / [ sum_i w_t^i ]

    The inner sum at each t runs only over trajectories with at least t+1
    timesteps -- shorter trajectories simply drop out once they end (no
    phantom zero-reward tail), which is the standard convention and why
    this estimator is generally lower-variance than trajectory-level WIS
    for the kind of variable-length, hourly ICU trajectories in this
    project.

    IMPORTANT, verified by direct construction (see the module's smoke
    tests): because trajectories have DIFFERENT lengths, this estimator
    will NOT numerically match compute_trajectory_wis / the plain
    per-trajectory-mean baseline even when the target and behavior
    policies are identical (all log-ratios zero). That's expected, not a
    bug -- at each timestep t, the denominator normalizes only over
    trajectories still active at t (shrinking as shorter trajectories end),
    not over all N trajectories, which is the standard, literature-
    accepted behavior of weighted per-decision IS. Don't read a
    trajectory-vs-stepwise gap as evidence of a computation error on its
    own; use the bootstrap CIs and the ESS curve (plot_stepwise_ess_curve)
    to judge whether either estimate is trustworthy at the horizons that
    matter.

    Also returns the per-timestep value and ESS curves, since these are
    useful diagnostics in their own right (how much of the horizon is
    still well-supported by a meaningful effective sample size).
    """
    max_len = max(t["length"] for t in trajectories)
    n_traj = len(trajectories)

    cum_log_ratio = np.full((n_traj, max_len), -np.inf)
    reward_matrix = np.zeros((n_traj, max_len))
    for i, t in enumerate(trajectories):
        L = t["length"]
        cum_log_ratio[i, :L] = np.cumsum(t["log_ratio"])
        reward_matrix[i, :L] = t["rewards"]

    value = 0.0
    per_step_value = np.zeros(max_len)
    per_step_ess = np.zeros(max_len)
    per_step_n_active = np.zeros(max_len, dtype=int)

    for t_idx in range(max_len):
        active = np.isfinite(cum_log_ratio[:, t_idx])
        n_active = int(active.sum())
        per_step_n_active[t_idx] = n_active
        if n_active == 0:
            continue

        lr_t = cum_log_ratio[active, t_idx]
        max_lr = lr_t.max()
        w_t = np.exp(lr_t - max_lr)
        r_t = reward_matrix[active, t_idx]

        step_value = np.sum(w_t * r_t) / np.sum(w_t)
        per_step_value[t_idx] = step_value
        per_step_ess[t_idx] = compute_ess(w_t)
        value += (gamma ** t_idx) * step_value

    return {
        "value": value,
        "per_step_value": per_step_value,
        "per_step_ess": per_step_ess,
        "per_step_n_active": per_step_n_active,
        "max_len": max_len,
    }


def compute_clinician_baseline_return(trajectories, gamma):
    """
    Mean +/- SEM discounted return under the REAL, logged clinician policy
    (no importance weighting at all) -- the natural reference point for
    interpreting the scale of a WIS estimate, in the same
    mean +/- SEM convention used elsewhere in this project (e.g. estimate_v0).
    """
    returns = discounted_returns(trajectories, gamma)
    return returns.mean(), returns.std() / np.sqrt(len(returns))


def summarize_weight_diagnostics(weights, top_k_fracs=(0.01, 0.05, 0.10)):
    """
    Weight-concentration diagnostics for a set of (trajectory-level)
    importance weights: effective sample size (raw and as a fraction of n),
    and how much of the total weight sits in the top 1/5/10% of
    trajectories by weight. A WIS estimate dominated by a handful of
    trajectories (low ESS fraction, high top-1% weight fraction) should be
    trusted less, regardless of its point value.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    total = w.sum()
    sorted_w = np.sort(w)[::-1]

    diag = {
        "n": n,
        "ess": compute_ess(w),
        "ess_frac": compute_ess(w) / n,
        "max_weight_frac": sorted_w[0] / total,
    }
    for frac in top_k_fracs:
        k = max(1, int(np.ceil(frac * n)))
        diag[f"top_{int(round(frac * 100))}pct_weight_frac"] = sorted_w[:k].sum() / total
    return diag


# ---------------------------------------------------------------------------
# Dominant-trajectory diagnostics -- for when summarize_weight_diagnostics
# says a low ESS / high top-1% weight fraction, and you need to find out
# WHY a trajectory-level WIS estimate is dominated by one (or a few)
# trajectories, not just THAT it is.
# ---------------------------------------------------------------------------

def find_dominant_trajectories(trajectories, traj_wis_result, top_k=5):
    """
    Ranks trajectories by their trajectory-level importance weight (from
    compute_trajectory_wis's returned dict) and, for each of the top_k,
    finds the single timestep WITHIN that trajectory with the largest
    per-step log importance ratio -- the specific clinical decision most
    responsible for that trajectory dominating the whole-episode estimate.

    Requires trajectories built with build_trajectory_data (needs each
    entry's start/behavior_lp/target_lp fields).

    Returns a list of summary dicts, ordered by descending weight.
    """
    weights = traj_wis_result["weights"]
    returns = traj_wis_result["returns"]
    total_weight = weights.sum()
    order = np.argsort(weights)[::-1][:top_k]

    summaries = []
    for rank, idx in enumerate(order):
        t = trajectories[idx]
        dominant_t = int(np.argmax(t["log_ratio"]))
        summaries.append({
            "rank": rank,
            "stay_id": t["stay_id"],
            "weight": weights[idx],
            "weight_frac": weights[idx] / total_weight,
            "discounted_return": returns[idx],
            "length": t["length"],
            "dominant_timestep_within_traj": dominant_t,
            "dominant_timestep_global_row": t["start"] + dominant_t,
            "dominant_log_ratio": t["log_ratio"][dominant_t],
            "dominant_behavior_lp": t["behavior_lp"][dominant_t],
            "dominant_target_lp": t["target_lp"][dominant_t],
            "trajectory_total_log_weight": t["log_ratio"].sum(),
        })
    return summaries


def print_dominant_trajectory_report(
        summaries, processor, bc_actor,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    Human-readable report for each dominant-trajectory summary from
    find_dominant_trajectories: the REAL logged (fluid, vaso) action at
    the dominant timestep, and BC's fluid/vaso fire probability at that
    same state -- so you can see directly whether this is a genuine, rare
    extreme dose BC was never going to model well, or something else
    worth digging into (e.g. a state where BC's fluid gate miscalibration,
    already documented in bc_with_logprobs.py, is at its worst).
    """
    states = torch.as_tensor(_to_numpy(processor.rl_dataset["states"]), dtype=torch.float32)
    actions = torch.as_tensor(_to_numpy(processor.rl_dataset["actions"]), dtype=torch.float32)

    bc_actor.eval()
    for s in summaries:
        row = s["dominant_timestep_global_row"]
        state = states[row:row + 1].to(device)
        action = actions[row:row + 1]
        fluid_dose, vaso_dose = action[0, 0].item(), action[0, 1].item()

        with torch.no_grad():
            _, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = bc_actor(
                state, return_dist=True,
            )

        print(f"--- rank {s['rank']}: stay_id={s['stay_id']}, "
              f"weight_frac={100 * s['weight_frac']:.1f}%, length={s['length']} ---")
        print(f"  Dominant timestep: {s['dominant_timestep_within_traj']} of {s['length']} "
              f"(global row {row})")
        print(f"  Real logged action: fluid={fluid_dose:.4f}, vaso={vaso_dose:.4f}")
        print(f"  BC fire probability at this state: fluid_p={f_p.item():.4f}, vaso_p={v_p.item():.4f}")
        print(f"  behavior_lp={s['dominant_behavior_lp']:.3f}, target_lp={s['dominant_target_lp']:.3f}, "
              f"log_ratio={s['dominant_log_ratio']:.3f} (exp(log_ratio)={np.exp(s['dominant_log_ratio']):.3e})")
        print(f"  Trajectory's total log-weight (sum over all {s['length']} timesteps): "
              f"{s['trajectory_total_log_weight']:.3f}")
        print(f"  Discounted return G_i for this trajectory: {s['discounted_return']:.4f}\n")


# ---------------------------------------------------------------------------
# Stepwise (cross-sectional) diagnostics -- for when plot_stepwise_ess_curve
# shows the ESS/n_active ratio collapsing at a SPECIFIC timestep (e.g. t=0),
# rather than late in the horizon. find_dominant_trajectories above answers
# "which trajectory dominates the whole-episode weight"; this answers
# "which trajectory dominates the cross-section AT timestep t_idx", which is
# the question that actually matters for compute_stepwise_wis, since that
# estimator renormalizes independently at every t.
# ---------------------------------------------------------------------------

def diagnose_stepwise_timestep(trajectories, t_idx, top_k=5):
    """
    Cross-sectional analogue of find_dominant_trajectories, pinned to one
    absolute timestep t_idx (0-indexed, matching compute_stepwise_wis's
    per_step_* arrays) rather than ranking by whole-trajectory weight.

    Among trajectories still active at t_idx (length > t_idx), computes the
    CUMULATIVE log-ratio from step 0 through t_idx (this is exactly the
    quantity compute_stepwise_wis exponentiates to get w_t^i), ranks
    trajectories by that cumulative weight, and reports both the cumulative
    figure and the SINGLE-STEP log_ratio/behavior_lp/target_lp at t_idx
    itself -- so you can tell whether a trajectory dominates the t_idx
    cross-section because of something happening right at t_idx, or because
    of ratio build-up from earlier steps in the same trajectory.

    Also returns n_active and the ESS at t_idx (should match
    compute_stepwise_wis's per_step_ess[t_idx] up to the max-subtraction
    used there for numerical stability, which does not change ESS).

    Requires trajectories built with build_trajectory_data (needs each
    entry's start/behavior_lp/target_lp fields).
    """
    active_idx = [i for i, t in enumerate(trajectories) if t["length"] > t_idx]
    n_active = len(active_idx)
    if n_active == 0:
        raise ValueError(f"No trajectories are active at t_idx={t_idx} (all shorter).")

    cum_lr = np.array([trajectories[i]["log_ratio"][:t_idx + 1].sum() for i in active_idx])
    max_lr = cum_lr.max()
    w = np.exp(cum_lr - max_lr)
    ess_at_t = compute_ess(w)
    total_w = w.sum()

    order = np.argsort(w)[::-1][:top_k]
    summaries = []
    for rank, k in enumerate(order):
        i = active_idx[k]
        t = trajectories[i]
        summaries.append({
            "rank": rank,
            "stay_id": t["stay_id"],
            "t_idx": t_idx,
            "n_active_at_t": n_active,
            "ess_at_t": ess_at_t,
            "weight_frac_at_t": w[k] / total_w,
            "cumulative_log_ratio_0_to_t": cum_lr[k],
            "global_row": t["start"] + t_idx,
            "step_log_ratio": t["log_ratio"][t_idx],
            "step_behavior_lp": t["behavior_lp"][t_idx],
            "step_target_lp": t["target_lp"][t_idx],
            "trajectory_length": t["length"],
        })
    return summaries


def print_stepwise_timestep_report(
        summaries, processor, bc_actor,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    Human-readable report for diagnose_stepwise_timestep's output: the real
    logged (fluid, vaso) action and BC's fire probability at the specific
    global row for each top-ranked trajectory, so you can see directly
    whether the t_idx cross-section is dominated by a genuine rare/extreme
    early dose, or by ratio build-up carried in from earlier steps within
    the same trajectory (compare step_log_ratio to
    cumulative_log_ratio_0_to_t: if they're close, the step itself is doing
    the damage; if cumulative is much more negative/positive than the step
    alone, earlier steps in that trajectory are responsible).
    """
    states = torch.as_tensor(_to_numpy(processor.rl_dataset["states"]), dtype=torch.float32)
    actions = torch.as_tensor(_to_numpy(processor.rl_dataset["actions"]), dtype=torch.float32)

    bc_actor.eval()
    for s in summaries:
        row = s["global_row"]
        state = states[row:row + 1].to(device)
        action = actions[row:row + 1]
        fluid_dose, vaso_dose = action[0, 0].item(), action[0, 1].item()

        with torch.no_grad():
            _, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = bc_actor(
                state, return_dist=True,
            )

        print(f"--- rank {s['rank']}: stay_id={s['stay_id']}, t_idx={s['t_idx']}, "
              f"weight_frac_at_t={100 * s['weight_frac_at_t']:.1f}% "
              f"(of {s['n_active_at_t']} active, ess_at_t={s['ess_at_t']:.2f}) ---")
        print(f"  Global row {row} (trajectory length {s['trajectory_length']})")
        print(f"  Real logged action: fluid={fluid_dose:.4f}, vaso={vaso_dose:.4f}")
        print(f"  BC fire probability at this state: fluid_p={f_p.item():.4f}, vaso_p={v_p.item():.4f}")
        print(f"  THIS STEP: behavior_lp={s['step_behavior_lp']:.3f}, target_lp={s['step_target_lp']:.3f}, "
              f"log_ratio={s['step_log_ratio']:.3f}")
        print(f"  CUMULATIVE (0..t_idx): log_ratio={s['cumulative_log_ratio_0_to_t']:.3f} "
              f"(exp={np.exp(s['cumulative_log_ratio_0_to_t']):.3e})\n")


# ---------------------------------------------------------------------------
# Gate vs. magnitude decomposition of log_prob_for_wis -- log_prob_for_wis
# itself (cql_with_logprobs.py) does NOT expose return_components like
# log_prob_of_action does, so this reimplements its exact arithmetic
# (Bernoulli gate log-prob + BC-bandwidth Gaussian-in-logit magnitude
# log-prob per drug) to split out the two terms for a single state/action
# pair. Use this to test the hypothesis that extreme negative per-step
# log-ratios come from the magnitude term (unbounded -- can blow up when
# BC's fitted std at that state is tiny and CQL's mean sits many std's away
# from the real logit-dose) rather than the gate term (bounded below by the
# 1e-8 floor inside each log(), i.e. at most ~-18.4 per drug, ~-36.8 total).
# ---------------------------------------------------------------------------

def decompose_log_prob_for_wis(cql_actor, bc_actor, state, action):
    """
    Recomputes cql_actor.log_prob_for_wis(state, action, bc_actor) for a
    single (state, action) pair (batch dim 1), returning a dict with the
    gate and magnitude log-prob for each drug separately, plus the
    normalized distance (in BC-std units) between CQL's mean and the real
    logit-dose that drives the magnitude term -- the direct numerical
    signature of the "CQL mean far from real action, BC std small at this
    state" failure mode.

    state, action: torch tensors, already on the same device as the models,
    shape (1, state_dim) / (1, 2).
    """
    bc_actor.eval()
    cql_actor.eval()

    with torch.no_grad():
        f_gate_logit, v_gate_logit, f_raw_mean, v_raw_mean, _, _ = cql_actor._dists(state)
        f_p = torch.sigmoid(f_gate_logit)
        v_p = torch.sigmoid(v_gate_logit)

        _, _, _, (_, _, _, _, bc_f_std, bc_v_std) = bc_actor(state, return_dist=True)

        f_dose, v_dose = action[:, 0:1], action[:, 1:2]
        f_fired_actual = (f_dose > cql_actor.zero_threshold).float()
        v_fired_actual = (v_dose > cql_actor.zero_threshold).float()

        f_gate_lp = f_fired_actual * torch.log(f_p + 1e-8) + (1 - f_fired_actual) * torch.log(1 - f_p + 1e-8)
        v_gate_lp = v_fired_actual * torch.log(v_p + 1e-8) + (1 - v_fired_actual) * torch.log(1 - v_p + 1e-8)

        f_a = f_dose.clamp(1e-4, 1 - 1e-4)
        v_a = v_dose.clamp(1e-4, 1 - 1e-4)
        f_logit_dose = torch.logit(f_a)
        v_logit_dose = torch.logit(v_a)

        f_dist = torch.distributions.Normal(f_raw_mean, bc_f_std)
        v_dist = torch.distributions.Normal(v_raw_mean, bc_v_std)

        f_mag_lp = f_dist.log_prob(f_logit_dose) - torch.log(f_a * (1 - f_a) + 1e-6)
        v_mag_lp = v_dist.log_prob(v_logit_dose) - torch.log(v_a * (1 - v_a) + 1e-6)

        f_z = (f_logit_dose - f_raw_mean) / bc_f_std
        v_z = (v_logit_dose - v_raw_mean) / bc_v_std

        total_lp = f_gate_lp + f_fired_actual * f_mag_lp + v_gate_lp + v_fired_actual * v_mag_lp

    return {
        "total_lp": total_lp.item(),
        "fluid_fired_actual": bool(f_fired_actual.item()),
        "vaso_fired_actual": bool(v_fired_actual.item()),
        "fluid_gate_lp": f_gate_lp.item(),
        "vaso_gate_lp": v_gate_lp.item(),
        "fluid_mag_lp": f_mag_lp.item(),
        "vaso_mag_lp": v_mag_lp.item(),
        "fluid_z": f_z.item(),      # (real logit-dose - CQL mean) / BC std
        "vaso_z": v_z.item(),
        "fluid_bc_std": bc_f_std.item(),
        "vaso_bc_std": bc_v_std.item(),
    }


def compare_dominant_trajectory_across_seeds(results, top_k=1):
    """
    For each seed's results (from run_wis_across_seeds, which stores each
    seed's own trajectories/trajectory_wis/bc_actor), finds the single
    most dominant trajectory by weight and reports its stay_id side by
    side across seeds.

    If the SAME stay_id dominates multiple seeds, that points to a
    structural property of that trajectory (e.g. an action BC's behavior
    density assigns very low probability to regardless of BC seed) rather
    than ordinary sampling noise -- worth knowing before deciding whether
    weight capping alone is an adequate fix, or whether that specific
    trajectory/action deserves its own look.
    """
    rows = []
    for r in results:
        summaries = find_dominant_trajectories(r["trajectories"], r["trajectory_wis"], top_k=top_k)
        top = summaries[0]
        rows.append({
            "seed": r["seed"],
            "stay_id": top["stay_id"],
            "weight_frac": top["weight_frac"],
            "discounted_return": top["discounted_return"],
        })

    print(f"{'seed':>5} {'stay_id':>12} {'weight_frac':>12} {'return':>10}")
    for row in rows:
        print(f"{row['seed']:>5} {str(row['stay_id']):>12} "
              f"{100 * row['weight_frac']:>11.1f}% {row['discounted_return']:>10.4f}")

    stay_ids = [row["stay_id"] for row in rows]
    if len(set(stay_ids)) == 1:
        print(f"\nSame stay_id ({stay_ids[0]}) dominates all {len(rows)} seeds -- "
              f"this looks like a structural property of that trajectory, not per-seed noise.")
    elif len(set(stay_ids)) < len(stay_ids):
        print(f"\n{len(set(stay_ids))} distinct stay_ids dominate across {len(rows)} seeds "
              f"(some repeats) -- partially structural, partially seed-dependent.")
    else:
        print(f"\nAll {len(rows)} seeds are dominated by different stay_ids -- "
              f"more consistent with ordinary per-seed sampling noise than a single problem trajectory.")

    return rows


def compute_length_weight_relationship(trajectories, traj_wis_result):
    """
    Tests whether trajectory-level WIS's weight concentration is a LENGTH
    artifact rather than a single-bad-decision artifact: log_weight for a
    trajectory is the SUM of its per-step log-ratios, so it compounds
    multiplicatively with trajectory length even when the average per-step
    disagreement between target and behavior policy is small and fairly
    constant. A long enough trajectory can dominate the whole WIS estimate
    on length alone, independent of whether any single decision in it was
    unusual (see find_dominant_trajectories: its "dominant timestep" can
    contribute a small fraction of a long trajectory's total log-weight,
    exactly this pattern).

    Returns per-trajectory arrays (lengths, log_weights,
    mean_log_ratio_per_step) plus the Spearman correlation between length
    and log_weight across ALL trajectories -- a strong positive
    correlation supports the length-compounding explanation over a
    small-number-of-bad-decisions explanation.
    """
    lengths = np.array([t["length"] for t in trajectories])
    log_weights = traj_wis_result["log_weights"]
    mean_log_ratio_per_step = np.array([t["log_ratio"].mean() for t in trajectories])

    rho, pval = stats.spearmanr(lengths, log_weights)
    print(f"Spearman correlation, trajectory length vs. log-weight: rho={rho:.3f} (p={pval:.2e}), n={len(lengths)}")
    print(f"Trajectory length: median={np.median(lengths):.0f}, max={lengths.max()}, "
          f"90th pctile={np.percentile(lengths, 90):.0f}")

    return {
        "lengths": lengths,
        "log_weights": log_weights,
        "mean_log_ratio_per_step": mean_log_ratio_per_step,
        "spearman_rho": rho,
        "spearman_pvalue": pval,
    }


def plot_length_vs_log_weight(length_weight_result, save_path=None):
    """
    Scatter of trajectory length vs. trajectory-level log-weight, colored
    by that trajectory's mean per-step log-ratio -- visualizes whether
    length alone (x-axis) predicts log-weight (y-axis) regardless of color
    (i.e. even trajectories with a middling, unremarkable average per-step
    log-ratio still reach an extreme total log-weight purely by being long).
    """
    lengths = length_weight_result["lengths"]
    log_weights = length_weight_result["log_weights"]
    mean_lr = length_weight_result["mean_log_ratio_per_step"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(lengths, log_weights, c=mean_lr, cmap="coolwarm", s=18, alpha=0.85,
                     vmin=-np.abs(mean_lr).max(), vmax=np.abs(mean_lr).max())
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("mean per-step log-ratio (target - behavior)")

    ax.set_xlabel("Trajectory length (timesteps)")
    ax.set_ylabel("Trajectory-level log-weight\n(sum of per-step log-ratios)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Trajectory length vs. WIS log-weight (Spearman "
        f"$\\rho$={length_weight_result['spearman_rho']:.2f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def bootstrap_wis_ci(trajectories, gamma, estimator="both", n_bootstrap=1000,
                      ci=(2.5, 97.5), random_state=None):
    """
    Bootstrap over TRAJECTORIES (patients), resampling with replacement and
    recomputing the chosen WIS estimator(s) each time. WIS has no simple
    closed-form variance (numerator and denominator are both random), so
    this is the standard approach for its uncertainty in the
    offline-RL-for-sepsis literature this project follows.

    estimator: "trajectory", "stepwise", or "both".
    Returns a dict keyed by estimator name, each with samples/mean/std/CI.
    """
    rng = np.random.default_rng(random_state)
    n = len(trajectories)
    want_traj = estimator in ("trajectory", "both")
    want_step = estimator in ("stepwise", "both")

    traj_samples, step_samples = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = [trajectories[i] for i in idx]
        if want_traj:
            traj_samples.append(compute_trajectory_wis(resampled, gamma)["value"])
        if want_step:
            step_samples.append(compute_stepwise_wis(resampled, gamma)["value"])

    def _summarize(vals):
        vals = np.array(vals)
        return {
            "samples": vals,
            "mean": vals.mean(),
            "std": vals.std(),
            "ci_lower": np.percentile(vals, ci[0]),
            "ci_upper": np.percentile(vals, ci[1]),
        }

    results = {}
    if want_traj:
        results["trajectory"] = _summarize(traj_samples)
    if want_step:
        results["stepwise"] = _summarize(step_samples)
    return results


# ---------------------------------------------------------------------------
# Orchestration across the 5 matched (CQL, BC) seed pairs
# ---------------------------------------------------------------------------

def run_wis_across_seeds(
        cql_models_with_seeds, bc_models_with_seeds, processor,
        gamma=0.975, cap_percentile=None, batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_bootstrap=1000, random_state=0,
    ):
    """
    Runs the full WIS pipeline (log-probs -> trajectory split -> both
    estimators -> bootstrap CI -> weight diagnostics) for every correctly
    seed-matched (CQL, BC) pair, printing a per-seed summary and returning
    the full per-seed results for further plotting/analysis.

    cql_models_with_seeds: list of (actor, critic, config_manager,
    hyperparams) tuples (model_type='cql'). bc_models_with_seeds: list of
    (actor, config_manager, hyperparams) tuples (model_type='bc'). Seed is
    read from each tuple's own hyperparams['seed'] -- see
    match_cql_bc_pairs's docstring for the matching logic and its noted
    caveats (no error on a duplicate or mismatched seed set).

    gamma should match the discount used for the reward type this
    processor's actions/rewards were built for (0.975 for health-score
    reward, per this project's convention -- confirm against your own
    config if evaluating a different reward type).
    """
    pairs, _, _ = match_cql_bc_pairs(cql_models_with_seeds, bc_models_with_seeds)
    states = processor.rl_dataset["states"]
    actions = processor.rl_dataset["actions"]

    results = []
    for cql_model_entry, bc_model_entry, seed in pairs:
        print(f"\n=== Seed {seed} ===")
        cql_actor = get_cql_actor(cql_model_entry)
        bc_actor = get_bc_actor(bc_model_entry)

        behavior_lp, target_lp = compute_behavior_and_target_log_probs(
            cql_actor, bc_actor, states, actions, batch_size=batch_size, device=device,
        )
        trajectories, cap_bounds = build_trajectory_data(
            processor, behavior_lp, target_lp, cap_percentile=cap_percentile,
        )

        traj_wis = compute_trajectory_wis(trajectories, gamma)
        step_wis = compute_stepwise_wis(trajectories, gamma)
        baseline_mean, baseline_sem = compute_clinician_baseline_return(trajectories, gamma)
        ci = bootstrap_wis_ci(
            trajectories, gamma, estimator="both",
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        diag = summarize_weight_diagnostics(traj_wis["weights"])

        print(f"  Clinician baseline (logged) return: {baseline_mean:.4f} +/- {baseline_sem:.4f} (SEM)")
        print(f"  Trajectory-level WIS: {traj_wis['value']:.4f}  "
              f"[{ci['trajectory']['ci_lower']:.4f}, {ci['trajectory']['ci_upper']:.4f}]  "
              f"ESS={traj_wis['ess']:.1f}/{traj_wis['n_trajectories']} "
              f"({100 * diag['ess_frac']:.1f}%)")
        print(f"  Stepwise WIS:         {step_wis['value']:.4f}  "
              f"[{ci['stepwise']['ci_lower']:.4f}, {ci['stepwise']['ci_upper']:.4f}]")
        print(f"  Top-1% of trajectories hold {100 * diag['top_1pct_weight_frac']:.1f}% of total weight "
              f"(max single trajectory: {100 * diag['max_weight_frac']:.1f}%)")

        results.append({
            "seed": seed,
            "trajectories": trajectories,
            "cap_bounds": cap_bounds,
            "trajectory_wis": traj_wis,
            "stepwise_wis": step_wis,
            "baseline_mean": baseline_mean,
            "baseline_sem": baseline_sem,
            "bootstrap_ci": ci,
            "weight_diagnostics": diag,
            "n_trajectories": len(trajectories),
            # kept for post-hoc diagnostics (e.g. print_dominant_trajectory_report,
            # compare_dominant_trajectory_across_seeds) without needing to re-derive
            # which BC/CQL actor belongs to this seed afterward
            "bc_actor": bc_actor,
            "cql_actor": cql_actor,
        })

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_wis_by_seed(results, title="", save_path=None):
    """
    Point-and-error-bar plot of trajectory-level and stepwise WIS estimates
    (with bootstrap CIs) across seeds, against the observed clinician
    baseline return (mean +/- SEM).
    """
    seeds = [r["seed"] for r in results]
    x = np.arange(len(seeds))

    traj_vals = np.array([r["trajectory_wis"]["value"] for r in results])
    traj_lo = np.array([r["bootstrap_ci"]["trajectory"]["ci_lower"] for r in results])
    traj_hi = np.array([r["bootstrap_ci"]["trajectory"]["ci_upper"] for r in results])

    step_vals = np.array([r["stepwise_wis"]["value"] for r in results])
    step_lo = np.array([r["bootstrap_ci"]["stepwise"]["ci_lower"] for r in results])
    step_hi = np.array([r["bootstrap_ci"]["stepwise"]["ci_upper"] for r in results])

    baseline_mean = np.mean([r["baseline_mean"] for r in results])
    baseline_sem = np.mean([r["baseline_sem"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    offset = 0.15
    ax.errorbar(x - offset, traj_vals, yerr=[traj_vals - traj_lo, traj_hi - traj_vals],
                fmt="o", color=TRAJECTORY_COLOR, capsize=4, label="Trajectory-level WIS")
    ax.errorbar(x + offset, step_vals, yerr=[step_vals - step_lo, step_hi - step_vals],
                fmt="s", color=STEPWISE_COLOR, capsize=4, label="Stepwise (per-decision) WIS")

    ax.axhline(baseline_mean, color=BASELINE_COLOR, linewidth=1.2, linestyle="--",
               label="Clinician (logged) baseline")
    ax.fill_between([-0.5, len(seeds) - 0.5],
                     baseline_mean - baseline_sem, baseline_mean + baseline_sem,
                     color=BASELINE_COLOR, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Estimated policy value (discounted return)")
    ax.set_xlim(-0.5, len(seeds) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title or "WIS-estimated CQL policy value by seed (bootstrap 95% CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_bootstrap_distributions(results, save_path=None):
    """Histogram of bootstrap WIS samples per seed, for both estimators."""
    n_seeds = len(results)
    fig, axes = plt.subplots(2, n_seeds, figsize=(3.2 * n_seeds, 6), sharey="row")
    if n_seeds == 1:
        axes = axes.reshape(2, 1)

    for col, r in enumerate(results):
        traj_samples = r["bootstrap_ci"]["trajectory"]["samples"]
        step_samples = r["bootstrap_ci"]["stepwise"]["samples"]

        axes[0, col].hist(traj_samples, bins=40, color=TRAJECTORY_COLOR, alpha=0.8)
        axes[0, col].axvline(r["trajectory_wis"]["value"], color="black", linewidth=1)
        axes[0, col].set_title(f"seed {r['seed']}\ntrajectory-level", fontsize=9)
        axes[0, col].spines[["top", "right"]].set_visible(False)

        axes[1, col].hist(step_samples, bins=40, color=STEPWISE_COLOR, alpha=0.8)
        axes[1, col].axvline(r["stepwise_wis"]["value"], color="black", linewidth=1)
        axes[1, col].set_title("stepwise", fontsize=9)
        axes[1, col].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Bootstrap distributions of WIS estimates (1000 resamples of trajectories)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_stepwise_ess_curve(results, save_path=None):
    """
    Per-timestep effective sample size (as a fraction of trajectories still
    active at that timestep) across the trajectory horizon, one line per
    seed -- shows how quickly the stepwise estimator's effective support
    erodes as importance weights compound over longer horizons.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax2 = ax1.twinx()

    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        step = r["stepwise_wis"]
        n_active = step["per_step_n_active"]
        ess_frac = np.divide(step["per_step_ess"], n_active,
                              out=np.zeros_like(step["per_step_ess"]), where=n_active > 0)
        t = np.arange(step["max_len"])
        color = cmap(i / max(1, len(results) - 1))
        ax1.plot(t, ess_frac, color=color, label=f"seed {r['seed']}", linewidth=1.5)

    n_active_ref = results[0]["stepwise_wis"]["per_step_n_active"]
    ax2.plot(np.arange(len(n_active_ref)), n_active_ref, color=BASELINE_COLOR,
              linestyle=":", linewidth=1.2, label="# trajectories still active")

    ax1.set_xlabel("Timestep (hours into trajectory)")
    ax1.set_ylabel("ESS / # active trajectories at that timestep")
    ax2.set_ylabel("# trajectories still active", color=BASELINE_COLOR)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax1.set_title("Stepwise WIS effective sample size over the trajectory horizon",
                   fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_weight_diagnostics(results, save_path=None):
    """
    Bar chart of trajectory-level weight-concentration diagnostics per
    seed: ESS fraction, and how much of total weight sits in the top 1%
    of trajectories.
    """
    seeds = [r["seed"] for r in results]
    x = np.arange(len(seeds))
    ess_frac = [r["weight_diagnostics"]["ess_frac"] * 100 for r in results]
    top1_frac = [r["weight_diagnostics"]["top_1pct_weight_frac"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    ax.bar(x - width / 2, ess_frac, width, color=ESS_COLOR, label="ESS (% of n)")
    ax.bar(x + width / 2, top1_frac, width, color=WEIGHT_COLOR,
           label="Weight held by top 1% of trajectories (%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Trajectory-level importance-weight concentration by seed",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig