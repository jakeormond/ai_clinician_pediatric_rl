"""
src/rl/analysis/model_selection_from_hyperparameter_sweeps.py

Pick the best run of a hyperparameter sweep, for any model type (Dynamic-DeepHit, BC, SARSA, CQL, ...).

The criterion is fixed by the runs themselves and nothing else:
    the run's own validation score of its own loss metric, as saved at the end of training:
        results['hyperparameters']['loss_metric']                  -> the metric's name
        results['val_scores'][f"val/{loss_metric}"]                -> the final validation value
    The best run is the one with the best final validation value. Whether best means highest
    (a concordance index) or lowest (a loss) is set per metric in METRIC_HIGHER_IS_BETTER, or
    passed as higher_is_better=; an unknown metric is refused, never guessed. No test data is
    read and no model is loaded.

Runs are found and read with runs.py, so a sweep can sit in any folder layout (one folder
per configuration, one per seed, nested): every folder holding a results.json is a run.

Outputs, written next to the sweep: sorted_runs.csv (every run, best first) and
best_results.json (a copy of the best run's results.json).

Usage (notebook):
    import src.rl.analysis.model_selection_from_hyperparameter_sweeps as ms
    table = ms.sweep_table(sweep_dir)
    best = ms.save_selection(table, sweep_dir)
"""

from __future__ import annotations

import os
import shutil
import warnings
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.rl.analysis import runs as R

FAMILY = "sweep"

# True = the metric is to be maximised, False = minimised. Add the metrics your sweeps use here
# (or pass higher_is_better= to sweep_table); a metric not listed is refused, never guessed.
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "weighted_multi_C": True,          # time-dependent concordance index (Dynamic-DeepHit)
}

# A family that only reads results.json: selection needs no model family or data identity,
# so the same code serves supervised and RL sweeps alike.
R.register_family(R.FamilySpec(
    name=FAMILY,
    detect=lambda run_dir: False,      # requested by name, never auto-detected
    data_identity=lambda record, cm: R.DataIdentity(None, None, None, None, None),
))


LIST_REDUCTIONS = {"last": lambda v: v[-1], "max": max, "min": min}


def _val_score(record: R.RunRecord, metric: str, list_value: str = "last") -> float:
    scores = R.read_results_json(record.run_dir).get("val_scores", {})
    key = f"val/{metric}"
    if key not in scores:
        warnings.warn(f"{record.run_dir}: no '{key}' in results.json['val_scores']; run is left out of the ranking")
        return np.nan
    v = scores[key]
    if isinstance(v, (list, tuple)):
        if list_value not in LIST_REDUCTIONS:
            raise ValueError(f"list_value must be one of {sorted(LIST_REDUCTIONS)}, got {list_value!r}")
        if not v:
            return np.nan
        v = LIST_REDUCTIONS[list_value](v)
    return float(v)


def _varying_hyperparameters(records, ignore=R.DEFAULT_IGNORE_KEYS) -> pd.DataFrame:
    """Hyperparameters whose effective value differs between the runs (what the sweep varied)."""
    keys = sorted({k for r in records for k in r.effective} - set(ignore) - {"loss_metric"})
    df = pd.DataFrame({k: [str(r.effective.get(k, "MISSING")) for r in records] for k in keys})
    return df.loc[:, df.nunique() > 1]


def sweep_table(root, metric: Optional[str] = None, higher_is_better: Optional[bool] = None,
                list_value: str = "last",
                include: Optional[Callable[[R.RunRecord], bool]] = None) -> pd.DataFrame:
    """
    One row per run under `root`, best first: run (path relative to root), seed, the validation
    value of the loss metric, its rank, and the hyperparameters that differ between runs.

    metric           : name of the metric; default is each run's hyperparameters['loss_metric'] (all runs
                       must agree). Its validation value is results['val_scores'][f"val/{metric}"].
    higher_is_better : True to maximise the metric, False to minimise it; default looks the metric up in
                       METRIC_HIGHER_IS_BETTER.
    list_value       : only for runs that save the validation score as a list (e.g. one value per epoch):
                       "last" (default), "max" or "min" of that list is the run's score. If training restores
                       the best epoch, the saved model's score is the best value, not the last. A list with
                       more than one value is reported, so this is never applied unnoticed.
    include          : optional filter on RunRecords (e.g. lambda r: r.get("seed") == 0).
    """
    records = R.discover_runs(root, family=FAMILY, include=include)
    if not records:
        raise FileNotFoundError(f"no runs (folders with a results.json) under {root}")

    if metric is None:
        metrics = {r.hyperparams.get("loss_metric") for r in records}
        if len(metrics) != 1 or None in metrics:
            raise ValueError(f"the runs must all record one loss_metric in their hyperparameters (or pass metric=); "
                             f"found {sorted(map(str, metrics))}")
        metric = metrics.pop()
    if higher_is_better is None:
        if metric not in METRIC_HIGHER_IS_BETTER:
            raise ValueError(f"direction of '{metric}' is unknown: pass higher_is_better= or add it to METRIC_HIGHER_IS_BETTER")
        higher_is_better = METRIC_HIGHER_IS_BETTER[metric]

    table = pd.DataFrame({
        "run": [os.path.relpath(r.run_dir, root) for r in records],
        "seed": [r.seed for r in records],
        metric: [_val_score(r, metric, list_value) for r in records],
    })
    table = pd.concat([table, _varying_hyperparameters(records)], axis=1)
    table.attrs.update(metric=metric, higher_is_better=higher_is_better)

    n_multi = sum(len(v) > 1 for r in records
                  if isinstance(v := R.read_results_json(r.run_dir).get("val_scores", {}).get(f"val/{metric}"), (list, tuple)))
    if n_multi:
        print(f"{n_multi} runs save val/{metric} as a list of several values; using list_value={list_value!r}")
    n_missing = int(table[metric].isna().sum())
    if n_missing == len(table):
        have = sorted(R.read_results_json(records[0].run_dir).get("val_scores", {}))
        raise ValueError(f"no run has 'val/{metric}' in results.json['val_scores']; the first run has: {have}")
    table = table.sort_values(metric, ascending=not higher_is_better, na_position="last", kind="stable")
    table.insert(0, "rank", np.where(table[metric].notna(), np.arange(1, len(table) + 1), np.nan))
    table = table.reset_index(drop=True)
    print(f"{len(records)} runs under {root}; ranked by final validation {metric} "
          f"({'higher' if higher_is_better else 'lower'} is better)"
          + (f"; {n_missing} without a score" if n_missing else ""))
    return table


def save_selection(table: pd.DataFrame, root) -> pd.Series:
    """
    Write sorted_runs.csv and best_results.json (copy of the best run's results.json) into `root`
    and return the best row of `table`.
    """
    if table["rank"].isna().all():
        raise ValueError("no run has a validation score")
    best = table.iloc[0]
    table.to_csv(os.path.join(root, "sorted_runs.csv"), index=False)
    shutil.copyfile(os.path.join(root, best["run"], "results.json"), os.path.join(root, "best_results.json"))
    metric = table.attrs["metric"]
    print(f"best run: {best['run']}  ({metric} = {best[metric]:.4f})")
    return best