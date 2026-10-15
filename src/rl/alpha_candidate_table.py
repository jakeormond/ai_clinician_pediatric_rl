"""
src/rl/alpha_candidate_table.py

Final-alpha candidates side by side, plus the "no degenerate seed" gate. Run on VAL.
Pure pandas: it joins the outputs of the other alpha-sweep files, it computes nothing from models.

  sweep = eval_cql.run_alpha_sweep_analysis(...)                      # per (alpha, seed): advantages, occupancy
  dm    = {name: dose_hist.dose_distribution_metrics(actions, clin, dim) for dim, name in [(0, "Fluid"), (1, "Vaso")]}

  gate = degenerate_seed_gate(dm["Fluid"], dm["Vaso"])                # per alpha: does any seed degenerate?
  first_clean_alpha(gate, max_eligible_alpha(sweep))                    # first alpha clean from there up to the eligible limit
  candidate_table(sweep, dm["Fluid"], dm["Vaso"], gate, [0.015, 0.02, 0.025, 0.029, 0.03, 0.035, 0.05])

  select_final_alpha_rule(sweep, dm['Fluid'], dm['Vaso'])           # (alpha, per-alpha check table)

The gate was added after inspecting the dose distributions; report it as a post-hoc criterion.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def degenerate_seed_gate(fluid: pd.DataFrame, vaso: pd.DataFrame,
                         fluid_max_fire: float = 0.99, vaso_min_fire: float = 0.001) -> pd.DataFrame:
    """
    Per alpha, which seeds have a degenerate gate: fluid given on >= fluid_max_fire of rows
    (the fluid gate has collapsed to "always"), or vaso given on <= vaso_min_fire of rows
    (collapsed to "never"). Columns: fluid_degenerate_seeds, vaso_degenerate_seeds (counts),
    max_fluid_fire, min_vaso_fire, clean (True when no seed is degenerate in either).
    """
    f = fluid.groupby("alpha")["fire_rate"].agg(max_fluid_fire="max", n_f=lambda s: int((s >= fluid_max_fire).sum()))
    v = vaso.groupby("alpha")["fire_rate"].agg(min_vaso_fire="min", n_v=lambda s: int((s <= vaso_min_fire).sum()))
    g = f.join(v, how="inner").rename(columns={"n_f": "fluid_degenerate_seeds", "n_v": "vaso_degenerate_seeds"})
    g["clean"] = (g.fluid_degenerate_seeds == 0) & (g.vaso_degenerate_seeds == 0)
    return g


def max_eligible_alpha(sweep: pd.DataFrame) -> float:
    """
    Largest swept alpha below the first alpha at which any seed's own-critic advantage is <= 0
    (cascade step 1). Alphas beyond it are already excluded, so the gate is not asked to hold there.
    """
    worst = sweep.groupby("alpha")["own_adv_norm"].min().sort_index()
    bad = worst[worst <= 0]
    if bad.empty:
        return float(worst.index.max())
    below = worst.index[worst.index < bad.index[0]]
    return float(below.max())


def first_clean_alpha(gate: pd.DataFrame, max_alpha: Optional[float] = None) -> Optional[float]:
    """
    Smallest alpha that is clean AND whose every larger swept alpha up to `max_alpha` is clean too
    (None if none). Pass max_alpha=max_eligible_alpha(sweep) so alphas that already fail the
    own-critic step do not count against the gate.
    """
    clean = gate["clean"].sort_index()
    if max_alpha is not None:
        clean = clean[clean.index <= max_alpha]
    ok = clean[::-1].cummin()[::-1]               # clean from here to the end of the range
    hits = ok[ok].index
    return float(hits[0]) if len(hits) else None


def select_final_alpha_rule(sweep: pd.DataFrame, fluid: pd.DataFrame, vaso: pd.DataFrame,
                            occupancy_min: float = 0.95, fluid_max_fire: float = 0.99,
                            vaso_min_fire: float = 0.001, neighbour_check: bool = False):
    """
    The final-alpha rule. Eligible range: swept alphas below the first alpha at which ANY seed's
    own-critic advantage is <= 0. Within it, an alpha is acceptable when
      - occupancy >= occupancy_min in every seed, and
      - no seed is degenerate (fluid fire rate < fluid_max_fire, vaso fire rate > vaso_min_fire),
    at that alpha AND at every larger eligible alpha (the policy never collapses from here upward,
    which does not depend on how finely the grid is sampled). Choose the acceptable alpha with the
    highest mean SARSA advantage (ties: lower alpha).
    neighbour_check=True adds a GRID-DEPENDENT requirement: the alpha's two adjacent swept alphas must
    also be acceptable-here (no knife-edge picks). It is stricter on a coarse grid and weaker on a fine
    one, so it is off by default; use it only as a documented sensitivity or a deliberate extra margin.
    Returns (alpha or None, table with one row per eligible alpha and every check).
    The same call, with the same arguments, is used for every reward model.
    """
    s = sweep.groupby("alpha").agg(own_min=("own_adv_norm", "min"), occ_min=("occupancy_frac", "min"),
                                   sarsa_adv_mean=("sarsa_adv_norm", "mean"))
    g = degenerate_seed_gate(fluid, vaso, fluid_max_fire, vaso_min_fire)
    t = s.join(g[["clean"]].rename(columns={"clean": "nondegenerate"}), how="inner").sort_index()
    t = t[t.index <= max_eligible_alpha(sweep)].copy()
    t["occ_ok"] = t.occ_min >= occupancy_min
    t["ok_here"] = t.occ_ok & t.nondegenerate
    t["ok_from_here_up"] = t.ok_here[::-1].cummin()[::-1].astype(bool)
    t["neighbours_ok"] = t.ok_here & t.ok_here.shift(1, fill_value=False) & t.ok_here.shift(-1, fill_value=False)
    t["acceptable"] = t.ok_from_here_up & (t.neighbours_ok if neighbour_check else True)
    ok = t[t.acceptable]
    if ok.empty:
        return None, t
    best = ok.sarsa_adv_mean.max()
    return float(ok[ok.sarsa_adv_mean >= best - 1e-12].index.min()), t


def candidate_table(sweep: pd.DataFrame, fluid: pd.DataFrame, vaso: pd.DataFrame, gate: pd.DataFrame,
                    alphas: Iterable[float], round_to: int = 3) -> pd.DataFrame:
    """
    One row per alpha, seed-mean (and worst seed where it matters):
      sarsa_adv        SARSA advantage, normalised (mean, min over seeds)
      own_adv>0        number of seeds whose own-critic advantage is positive
      occupancy        support fraction (mean, min)
      fluid_*, vaso_*  fire rate, W1 to the clinician, low-dose share (see dose_histogram_by_alpha)
      degenerate       seeds with a collapsed fluid/vaso gate (from degenerate_seed_gate)
    """
    alphas = [a for a in alphas if a in set(sweep.alpha)]
    s = sweep[sweep.alpha.isin(alphas)].groupby("alpha")
    out = pd.DataFrame({
        "sarsa_adv_mean": s["sarsa_adv_norm"].mean(), "sarsa_adv_min": s["sarsa_adv_norm"].min(),
        "own_adv>0 (seeds)": s["own_adv_norm"].apply(lambda x: int((x > 0).sum())),
        "occupancy_mean": s["occupancy_frac"].mean(), "occupancy_min": s["occupancy_frac"].min(),
    })
    for name, d in (("fluid", fluid), ("vaso", vaso)):
        g = d[d.alpha.isin(alphas)].groupby("alpha")
        out[f"{name}_fire_mean"] = g["fire_rate"].mean()
        out[f"{name}_w1"] = g["w1"].mean()
        out[f"{name}_low_share"] = g["low_share"].mean()
    out["fluid_fire_max"] = fluid[fluid.alpha.isin(alphas)].groupby("alpha")["fire_rate"].max()
    out["vaso_fire_min"] = vaso[vaso.alpha.isin(alphas)].groupby("alpha")["fire_rate"].min()
    out["degenerate_seeds"] = (gate.loc[out.index, "fluid_degenerate_seeds"] + gate.loc[out.index, "vaso_degenerate_seeds"])
    cols = ["sarsa_adv_mean", "sarsa_adv_min", "own_adv>0 (seeds)", "occupancy_mean", "occupancy_min",
            "fluid_fire_mean", "fluid_fire_max", "fluid_w1", "fluid_low_share",
            "vaso_fire_mean", "vaso_fire_min", "vaso_w1", "vaso_low_share", "degenerate_seeds"]
    return out[cols].round(round_to)