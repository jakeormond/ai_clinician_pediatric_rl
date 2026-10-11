# AI Clinician (Paediatric): offline RL for fluid and vasopressor dosing in the PICU

This repository adapts the [AI Clinician](https://www.nature.com/articles/s41591-018-0213-5) offline
reinforcement-learning framework, originally built for adult sepsis, to fluid and vasopressor dosing
for children with sepsis in the paediatric intensive care unit (PICU).

The project is funded by the BMA Foundation and run at Imperial College London (Department of
Bioengineering). Lead: Jake Ormond. Data harmonisation: Edison Liu.

> **Status:** research code. The models are not validated for clinical use and must not be used to
> guide treatment. See [Known limitations](#known-limitations).

---

## Overview

Paediatric ICU mortality is low (under 10%), so the sparse survival reward used for adults gives the
agent very little signal. Instead, a supervised survival model produces a dense **hourly health
score**, and the reward is the hour-to-hour change in that score.

```
hourly PICU data ──► Dynamic-DeepHit (transformer, competing risks: death / discharge)
                        │
                        ├── latent state (per hour) ───────────────► RL state
                        └── health score (per hour) ──► r_t = HS_{t+1} − HS_t ──► RL reward
                                                                        │
                       ┌────────────────────────────────────────────────┤
                       ▼                    ▼                           ▼
              SARSA (clinician critic)   CQL (policy)       BC (behaviour policy)
                       │                    │                           │
                       └──── evaluation: advantage, FQE, WIS, dose-response shape,
                             occupancy (data-support) checks ◄──────────┘
```

The pipeline has two stages:

1. **Dynamic-DeepHit** (`src/supervised_learning/`): a transformer trained on 72-hour windows to
   predict time to death or discharge. Its per-hour outputs give both the RL state (final-timestep
   latents) and the health score.
2. **Offline RL** (`src/rl/`): critics and policies trained on the sepsis cohort using the
   Dynamic-DeepHit states and health-score rewards, plus a set of evaluation and diagnostic analyses.

## Data

- **Source:** Children's Hospital of Orange County (CHOC) PICU data, under a data sharing agreement.
  **No data are included in this repository**, and the data cannot be shared.
- **Cohorts:**
  - Dynamic-DeepHit is trained on the full PICU cohort (about 8,500 admissions).
  - RL models are trained on the sepsis subset (about 1,600 stays).
  - Fixed train / validation / test splits by admission.
- **Timestep:** hourly.
- **Actions:** fluid intake and vasoactive-inotropic score (`intake`, `vis`), each normalised to
  [0, 1]. Both are zero-inflated; vasopressors are given at roughly 9% of hours.
- **Planned external validation:** harmonised London PICU datasets, and a collaboration with the
  University of Queensland.

<!-- TODO: describe where the processed shards live and how they are produced (data processing
     scripts), or link to the data-preparation repo/docs. -->

## Repository layout

The main parts referred to in this README (other files exist; this is not a full listing):

```
src/
├── config_manager.py                  ConfigManager, load_configuration (paths, environments)
├── data/
│   └── data_processor_for_shards.py   ShardProcessor: loading/normalising HDF5 shards, RL datasets
├── supervised_learning/
│   ├── models/
│   │   └── dynamic_deephit_transformer.py   TransformerDeepHitMulti + trainer + loss
│   ├── training.py                    training loop, create_processor_from_shards
│   ├── predict.py                     inference, health score, evaluation plots
│   ├── dynamic_deephit_evaluation.py  C-index, Brier score, ROC/PRC, ECE
│   └── evaluate_dynamic_deephit_final_seeds.py   analysis for the final 5-seed models
└── rl/
    ├── replay_buffer.py               datasets incl. DynamicDeepHitTransformerDataset
    ├── training.py                    shared RL training loop, checkpoints
    ├── models/
    │   ├── basic_q.py                 SARSA critic; also FQE (Q_Learner with a target policy)
    │   ├── cql_with_logprobs.py       CQL actor (hurdle model) + critic
    │   └── bc_with_logprobs.py        behaviour cloning (behaviour policy for WIS)
    ├── analysis/                      shared analysis framework (see below)
    │   ├── runs.py                    run discovery, hyperparameters, overrides, data identity
    │   ├── models.py                  loading any run into a standard LoadedModel
    │   └── data.py                    processors, per-row helpers, ensemble health score
    └── *.py / *.ipynb                 per-analysis modules and notebooks (see below)
```

<!-- TODO: check the location of the supervised-learning entry point (supervised_learning.py) and
     the RL training entry point, and add them here. -->

## Stage 1: Dynamic-DeepHit and the health score

- **Model:** causal transformer (CNN embedding stem, 4 encoder layers, d_model 128) over 72-hour
  windows, with two competing-risk heads (death, discharge) over 8 time bins. An auxiliary head
  predicts next-step physiology.
- **Health score:** `HS = 50 · (1 + Σ p(e, τ) · w(e, τ))`, scaled to [0, 100]. Discharge soon scores
  high and death soon scores low.
- **Two input variants**, trained with 5 seeds each:
  - `with_actions`: fluid and vasopressor doses are included in the inputs. Used for the RL
    **state**, because treatment history is part of the patient's state.
  - `without_actions`: treatment columns are removed (inside `DynamicDeepHitTransformerDataset`
    when `include_actions_in_state = 'False'`). Used to test whether treatment in the reward
    model's inputs drives the policy.
- **Seed-averaged reward:** the hourly reward from a single seed is noisy; seeds agree on its
  hour-to-hour changes only weakly. The health score is therefore averaged over the 5 seeds and
  saved as `<job folder>/<split>_predictions.h5`, so every RL seed trains on the same reward. RL seed
  *i* uses the latents of Dynamic-DeepHit seed *i* as its state. The file format is documented in
  `src/rl/analysis/data.py`.

### Training

```bash
# from the repository root
python -m src.<...>.supervised_learning \
    --hospital choc --model_type dynamic_deephit_transformer \
    --prediction_type discharge_mortality_tte --mode single_split --split_num 0 \
    --cohort all --unit PICU --filter_old True \
    --include_actions_in_state True        # or False for the action-free variant
```
<!-- TODO: replace with the exact command / PBS job script used for the final runs. -->

### Inference and analysis

Run `evaluate_dynamic_deephit_final_seeds.ipynb` from top to bottom. It:

1. discovers the runs (`make_ddh_study`);
2. loads the models (`load_ddh_models`);
3. generates train/val/test predictions;
4. evaluates discrimination and calibration with a stay-level bootstrap;
5. measures agreement between seeds;
6. builds the seed-averaged health score;
7. runs the vaso-onset event study.

Inference windows are built exactly as in training, by reusing the training Dataset's preprocessing,
and are checked against `Dataset.__getitem__` before every prediction run.

## Stage 2: offline RL

| Model | File | Role |
|---|---|---|
| SARSA | `models/basic_q.py` | Critic of the clinicians' own policy. Used for advantage estimates and dose-response analysis. |
| CQL | `models/cql_with_logprobs.py` | The policy. Hurdle actor: a fire/no-fire gate times a dose magnitude, separately for fluid and vasopressor. Uses automatic entropy tuning and a mean-saturation regulariser. |
| BC | `models/bc_with_logprobs.py` | Behaviour policy for weighted importance sampling. The fluid gate uses a class-weighted loss. |
| FQE | `models/basic_q.py` | Fitted Q evaluation of each CQL policy. Each FQE model is paired with the CQL model of the same seed. |

- **Discount:** γ = 0.975.
- **Seeds:** all final models use 5 seeds.
- **Hyperparameter selection:** a single-run search first, then 5 seeds at the chosen
  configuration. The CQL alpha is chosen by excluding unstable values and then picking the highest
  occupancy support.

**Evaluation and diagnostics:**

- advantage of the policy over clinicians, under both the SARSA and FQE critics;
- the shape of the Q(dose) curve, classified as increasing, decreasing, flat, concave or convex;
- **occupancy support**: whether a recommended dose has real precedent in training data at
  similar severity. This is the proposed clinical safety flag;
- WIS (in progress).

## Analysis conventions

- **One notebook per analysis.** The notebook runs commands and shows plots and tables. A `.py` file
  of the same name holds the functions.
- **Shared framework in `src/rl/analysis`.** Analyses use it rather than family-specific loading
  code:
  ```python
  from src.rl.analysis import runs, models, data
  study  = runs.make_study(root, family="cql", group_by=("cql_alpha",))
  loaded = models.load_study_models(study)          # {group: {seed: LoadedModel}}
  store  = data.DataStore();  proc = store.processor(loaded[key][seed], split="val")
  ```
  - **Model families:** `cql`, `sarsa` and `dynamic_deephit` are built in. Add others with
    `register_family` and `register_loader`.
  - **Seeds and hyperparameters** always come from each run's own `results.json` or checkpoint,
    never from `config_manager.pkl`, which can be shared across runs.
  - **Hardcoded training-time values** are recorded in `run_overrides.yaml` files next to the runs
    and applied automatically.
  - **Cluster paths** are remapped with `runs.set_path_mappings([...])`. Call it before loading any
    runs.
- **Notebook autoreload.** `runs.py` and `models.py` hold registries, a model cache and path
  mappings, so exclude them from autoreload, and restart the kernel after editing them:
  ```python
  %load_ext autoreload
  %autoreload 2
  %aimport -src.rl.analysis.runs
  %aimport -src.rl.analysis.models
  ```

## Known limitations

- **The vasopressor policy recommends zero vaso.** Trained CQL policies recommend no vasopressor in
  almost every state. This persists across several checks:
  - with treatment removed from the reward model's inputs;
  - across the CQL alpha sweep;
  - with a terminal mortality reward.

  A small bonus for matching clinicians' vaso doses restores vaso recommendations, which shows the
  model *can* express them. The most likely explanation is **confounding by indication**: vaso is
  given to children who are deteriorating in ways the recorded state does not fully capture. This
  cannot be resolved with observational data alone. A sensitivity analysis is planned, and so is
  testing the pipeline on adult data (MIMIC) and other paediatric datasets.
- **Extreme fluid doses.** About 2–4% of states receive near-maximal fluid recommendations. These
  come from monotonically increasing Q(fluid) curves, which are the least data-supported conclusions
  the critic reaches. The recommended mitigation is per-recommendation occupancy flagging.
- **Terminal-reward models.** CQL with a terminal reward produced degenerate all-or-nothing dosing
  and was not carried forward. The Monte Carlo terminal-reward critics are used only as a robustness
  check.

## Lessons recorded in this project

1. Single-seed comparisons are unreliable. Confirm conclusions across seeds.
2. Check seed and config pairing from each run's own files, not from shared pickles or list order.
3. A fix ported from one model to another is not automatically safe. Check that the problem it
   solves exists in the new model first.
4. Build inference inputs with the training preprocessing, and verify them against it.

## Requirements

<!-- TODO: Python version, environment file (conda/pip), key packages
     (torch, numpy, pandas, h5py, scikit-learn, matplotlib, seaborn, scipy, wandb, PyYAML). -->

## Citation and acknowledgements

<!-- TODO: paper(s) in preparation; funder acknowledgement wording; collaborators; licence. -->