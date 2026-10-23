"""
Does training longer (fixed 1000-epoch schedule) change the CQL policies and what the SARSA critics say about them?

compare_cql_policies : per seed, the advantage of the early-stopped CQL policy and of the 1000-epoch CQL policy over the
                       clinician, judged by each SARSA critic given, plus how different the two policies' actions are.
plot_cql_comparison  : the result.

SARSA-critic comparisons live in evaluate_sarsa_models.py.
"""
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.rl.analysis import data as D
from src.rl.evaluate_cql_alpha_sweep import actor_actions, check_judge_matches
from src.rl.evaluate_fqe_final_seeds import _by_seed, _pct_ci

ACTION_NAMES = ("fluid", "vaso")          # columns of the action array


def compare_cql_policies(old_cql, new_cql, judges: Dict[str, Dict], store, split: str = "val", n_boot: int = 500,
                         boot_seed: int = 0, batch_size: int = 4096, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    old_cql, new_cql : {seed: LoadedModel} (one group each; same alpha)
    judges           : {name: {seed: LoadedModel}} SARSA critics, e.g. {"sarsa_old": ..., "sarsa_new": ...}
    store            : data.DataStore; each seed's data comes from the old CQL model's data identity (checked for all models)

    Advantage = Q(s, policy action) - Q(s, clinician action), normalised by the IQR of that critic's Q at clinician
    actions. Returns (adv, actions):
      adv     : one row per (seed, judge): adv_old / adv_new (+ CIs, stay bootstrap), d_adv = new - old with a paired CI,
                frac_pos_old / frac_pos_new
      actions : one row per seed: mean fluid / vaso dose of the clinician, old and new policy, mean |old - new| and the
                correlation between the two policies' doses, per action dimension
    """
    old, new = _by_seed(old_cql), _by_seed(new_cql)
    J_ = {k: _by_seed(v) for k, v in judges.items()}
    seeds = sorted(set(old) & set(new) & set.intersection(*[set(v) for v in J_.values()]), key=str)
    skipped = (set(old) | set(new) | set.union(*[set(v) for v in J_.values()])) - set(seeds)
    if not seeds:
        raise KeyError("no seed present in all model sets")
    if skipped:
        warnings.warn(f"seeds {sorted(skipped, key=str)} are not in all model sets and were skipped")

    adv_rows, act_rows = [], []
    for seed in seeds:
        check_judge_matches(old[seed], new[seed])
        for j in J_.values():
            check_judge_matches(old[seed], j[seed])
        p = store.processor(old[seed], split)
        s, a_clin = D.states(p), D.actions(p)
        inv = np.unique(D.trajectory_ids(p), return_inverse=True)[1]
        J = int(inv.max()) + 1
        n = np.bincount(inv, minlength=J).astype(np.float64)
        idx = np.random.default_rng(boot_seed).integers(0, J, size=(n_boot, J), dtype=np.int32)
        den = n[idx].sum(1)
        a_old = actor_actions(old[seed], s, batch_size=batch_size)
        a_new = actor_actions(new[seed], s, batch_size=batch_size)

        arow = {"seed": seed}
        for d, nm in enumerate(ACTION_NAMES):
            arow.update({f"{nm}_clin": float(a_clin[:, d].mean()), f"{nm}_old": float(a_old[:, d].mean()),
                         f"{nm}_new": float(a_new[:, d].mean()),
                         f"{nm}_absdiff": float(np.abs(a_old[:, d] - a_new[:, d]).mean()),
                         f"{nm}_corr": float(np.corrcoef(a_old[:, d], a_new[:, d])[0, 1]) if a_old[:, d].std() > 0 and a_new[:, d].std() > 0 else np.nan})
        act_rows.append(arow)

        for jname, jm in J_.items():
            c = jm[seed]
            q_cl = c.q(s, a_clin, batch_size=batch_size, scaled=True)
            iqr = max(float(np.quantile(q_cl, 0.75) - np.quantile(q_cl, 0.25)), 1e-6)
            res = {}
            for tag, a_pi in (("old", a_old), ("new", a_new)):
                adv = c.q(s, a_pi, batch_size=batch_size, scaled=True) - q_cl
                boot = (np.bincount(inv, weights=adv, minlength=J)[idx].sum(1) / den) / iqr
                res[tag] = (float(adv.mean() / iqr), boot, float((adv > 0).mean()))
            row = {"seed": seed, "judge": jname}
            for tag in ("old", "new"):
                row[f"adv_{tag}"] = res[tag][0]
                row[f"adv_{tag}_lo"], row[f"adv_{tag}_hi"] = _pct_ci(res[tag][1])
                row[f"frac_pos_{tag}"] = res[tag][2]
            row["d_adv"] = res["new"][0] - res["old"][0]
            row["d_adv_lo"], row["d_adv_hi"] = _pct_ci(res["new"][1] - res["old"][1])
            adv_rows.append(row)
            if verbose:
                print(f"  seed {seed} judged by {jname}: advantage old {row['adv_old']:+.3f} -> new {row['adv_new']:+.3f} "
                      f"(d {row['d_adv']:+.3f} [{row['d_adv_lo']:+.3f}, {row['d_adv_hi']:+.3f}])")
    return pd.DataFrame(adv_rows), pd.DataFrame(act_rows)


def plot_cql_comparison(adv: pd.DataFrame, actions: pd.DataFrame, title: str = "", save_path=None):
    """Left: advantage of old / new policy per judge. Middle: new - old with paired CI. Right: mean doses."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    seeds = list(dict.fromkeys(adv.seed)); judges = list(dict.fromkeys(adv.judge))
    x = np.arange(len(seeds)); w = 0.8 / (2 * len(judges))

    ax = axes[0]; ax.axhline(0, color="grey", lw=1)
    for k, jn in enumerate(judges):
        d = adv[adv.judge == jn].set_index("seed").loc[seeds]
        for t, (mk, off) in enumerate((("o", 0), ("s", 1))):
            tag = ("old", "new")[t]
            pos = x + (2 * k + t - len(judges) + 0.5) * w
            y = d[f"adv_{tag}"].to_numpy()
            ax.errorbar(pos, y, yerr=[y - d[f"adv_{tag}_lo"], d[f"adv_{tag}_hi"] - y], fmt=mk, color=f"C{k}",
                        mfc=f"C{k}" if tag == "new" else "none", capsize=2, label=f"{jn}: {tag} policy")
    ax.set_title("advantage over clinician"); ax.set_ylabel("advantage / IQR(Q at clinician actions)"); ax.legend(frameon=False, fontsize=7)

    ax = axes[1]; ax.axhline(0, color="grey", lw=1)
    for k, jn in enumerate(judges):
        d = adv[adv.judge == jn].set_index("seed").loc[seeds]
        y = d.d_adv.to_numpy()
        ax.errorbar(x + (k - (len(judges) - 1) / 2) * 0.15, y, yerr=[y - d.d_adv_lo, d.d_adv_hi - y], fmt="o", color=f"C{k}", capsize=3, label=jn)
    ax.set_title("new - old policy (paired CI)"); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]; a = actions.set_index("seed").loc[seeds]
    for t, (tag, mk) in enumerate((("clin", "^"), ("old", "o"), ("new", "s"))):
        ax.plot(x + (t - 1) * 0.12, a[f"{ACTION_NAMES[0]}_{tag}"], mk, color=f"C{t + 3}", label=f"{tag}")
    ax.set_title(f"mean {ACTION_NAMES[0]} dose"); ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x, [str(s) for s in seeds]); ax.set_xlabel("seed")
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# Consistency of the recommended doses across seeds
# =============================================================================

def policy_consistency(model_sets: Dict[str, Dict], store, split: str = "val", on_thresh: Dict[str, float] = None,
                       batch_size: int = 4096) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    How much do different seeds of the same model set recommend the same doses in the same patient-hours?

    model_sets : {name: {seed: LoadedModel}}, e.g. {"early_stopped": cql_old, "1000_epochs": cql_new}
    on_thresh  : {"fluid": x, "vaso": y}; a dose above the threshold counts as "on" for the overlap measure
                 (default 0.01 for both; set it before looking at results).
    Seeds use different state embeddings, but the patient-hours are the same rows; this is checked through the
    clinician actions.

    Returns (pairs, per_seed, summary):
      pairs    : one row per (set, dim, seed pair): Pearson and Spearman correlation of the doses, mean |difference|,
                 Jaccard overlap of the "on" hours
      per_seed : one row per (set, dim, seed): mean dose and share of hours "on"
      summary  : one row per (set, dim): mean / min of the pairwise numbers, mean across-seed SD of the dose per hour,
                 and the share of hours where all seeds agree on on/off
    """
    from itertools import combinations
    from scipy.stats import spearmanr
    on_thresh = {**{nm: 0.01 for nm in ACTION_NAMES}, **(on_thresh or {})}
    pairs, per_seed, summary = [], [], []
    for sname, models in model_sets.items():
        m = _by_seed(models)
        seeds = sorted(m, key=str)
        acts, clin = {}, None
        for sd in seeds:
            p = store.processor(m[sd], split)
            a_c = D.actions(p)
            if clin is None:
                clin = a_c
            elif a_c.shape != clin.shape or not np.allclose(a_c, clin):
                raise ValueError(f"{sname}: seed {sd} has different rows (clinician actions) from seed {seeds[0]}; "
                                 f"doses cannot be compared hour by hour")
            acts[sd] = actor_actions(m[sd], D.states(p), batch_size=batch_size)
        for d, nm in enumerate(ACTION_NAMES):
            on = {sd: acts[sd][:, d] > on_thresh[nm] for sd in seeds}
            for sd in seeds:
                per_seed.append({"set": sname, "dim": nm, "seed": sd, "mean_dose": float(acts[sd][:, d].mean()),
                                 "share_on": float(on[sd].mean())})
            rows = []
            for i, j in combinations(seeds, 2):
                x, y = acts[i][:, d], acts[j][:, d]
                union = (on[i] | on[j]).sum()
                rows.append({"set": sname, "dim": nm, "seed_i": i, "seed_j": j,
                             "pearson": float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan,
                             "spearman": float(spearmanr(x, y)[0]),
                             "mean_abs_diff": float(np.abs(x - y).mean()),
                             "jaccard_on": float((on[i] & on[j]).sum() / union) if union else np.nan})
            pairs.extend(rows)
            pr = pd.DataFrame(rows)
            stack = np.stack([acts[sd][:, d] for sd in seeds])
            onm = np.stack([on[sd] for sd in seeds])
            summary.append({"set": sname, "dim": nm, "n_seeds": len(seeds),
                            "pearson_mean": pr.pearson.mean(), "pearson_min": pr.pearson.min(),
                            "spearman_mean": pr.spearman.mean(), "jaccard_on_mean": pr.jaccard_on.mean(),
                            "jaccard_on_min": pr.jaccard_on.min(),
                            "mean_sd_across_seeds": float(stack.std(0).mean()),
                            "all_agree_on_off": float((onm.all(0) | (~onm).all(0)).mean()),
                            "any_on": float(onm.any(0).mean())})
    return pd.DataFrame(pairs), pd.DataFrame(per_seed), pd.DataFrame(summary)


def plot_policy_consistency(pairs: pd.DataFrame, title: str = "", save_path=None):
    """Pairwise agreement between seeds, one dot per seed pair, for each action dimension and model set."""
    dims = list(dict.fromkeys(pairs.dim)); sets = list(dict.fromkeys(pairs.set))
    metrics = (("pearson", "Pearson correlation"), ("jaccard_on", "overlap of 'on' hours (Jaccard)"))
    fig, axes = plt.subplots(len(dims), len(metrics), figsize=(5 * len(metrics), 3.4 * len(dims)), constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for r, dn in enumerate(dims):
        for c, (mk, label) in enumerate(metrics):
            ax = axes[r, c]
            for k, sn in enumerate(sets):
                y = pairs[(pairs.dim == dn) & (pairs.set == sn)][mk].to_numpy()
                ax.plot(k + rng.uniform(-0.08, 0.08, len(y)), y, "o", color=f"C{k}", alpha=0.7)
                ax.hlines(np.nanmean(y), k - 0.2, k + 0.2, color="k", lw=2)
            ax.set_xticks(range(len(sets)), sets); ax.set_xlim(-0.5, len(sets) - 0.5); ax.set_ylim(0, 1.02)
            ax.set_title(f"{dn}: {label}", fontsize=10)
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def overlap_by_threshold(model_sets: Dict[str, Dict], store, dim: str = "vaso", thresholds=(0.005, 0.01, 0.02, 0.05, 0.1),
                         split: str = "val", batch_size: int = 4096) -> pd.DataFrame:
    """
    Do seeds agree on WHEN to give a dose, and on HOW MUCH when both give one? One row per (set, threshold):
      share_on          : mean share of hours a seed is "on" (dose > threshold)
      any_on            : share of hours where at least one seed is on
      jaccard_on        : mean pairwise overlap of the "on" hours
      jaccard_chance    : the same if seeds were independent with their own on-rates (a crude floor; severity drives all seeds)
      dose_corr_both_on : mean pairwise Pearson correlation of the dose, on hours where BOTH seeds of the pair are on
      n_both_on         : mean number of such hours
    """
    from itertools import combinations
    d = ACTION_NAMES.index(dim)
    rows = []
    for sname, models in model_sets.items():
        m = _by_seed(models); seeds = sorted(m, key=str)
        x, clin = {}, None
        for sd in seeds:
            p = store.processor(m[sd], split)
            a_c = D.actions(p)
            if clin is None:
                clin = a_c
            elif a_c.shape != clin.shape or not np.allclose(a_c, clin):
                raise ValueError(f"{sname}: seed {sd} has different rows from seed {seeds[0]}")
            x[sd] = actor_actions(m[sd], D.states(p), batch_size=batch_size)[:, d]
        for th in thresholds:
            on = {sd: x[sd] > th for sd in seeds}
            jac, chance, corr, nb = [], [], [], []
            for i, j in combinations(seeds, 2):
                union = (on[i] | on[j]).sum()
                jac.append((on[i] & on[j]).sum() / union if union else np.nan)
                pi, pj = on[i].mean(), on[j].mean()
                chance.append(pi * pj / (pi + pj - pi * pj) if (pi + pj) > 0 else np.nan)
                both = on[i] & on[j]
                nb.append(int(both.sum()))
                corr.append(np.corrcoef(x[i][both], x[j][both])[0, 1] if both.sum() > 2 and x[i][both].std() > 0 and x[j][both].std() > 0 else np.nan)
            rows.append({"set": sname, "threshold": th, "share_on": float(np.mean([on[s].mean() for s in seeds])),
                         "any_on": float(np.stack([on[s] for s in seeds]).any(0).mean()),
                         "jaccard_on": float(np.nanmean(jac)), "jaccard_chance": float(np.nanmean(chance)),
                         "dose_corr_both_on": float(np.nanmean(corr)) if not np.all(np.isnan(corr)) else np.nan,
                         "n_both_on": float(np.mean(nb))})
    return pd.DataFrame(rows)