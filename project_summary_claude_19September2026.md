# Pediatric Sepsis Offline RL Project — Summary

**Project**: Adapting the AI Clinician (offline RL treatment-recommendation framework, originally built for adult sepsis) to pediatric fluid/vasopressor dosing in the PICU. BMA Foundation–funded, Imperial College London. Lead: Jake Ormond (postdoctoral researcher), with PhD student Edison Liu on data harmonisation.

**Data**: Children's Hospital of Orange County (CHOC) PICU dataset.
- Full cohort (Dynamic DeepHit training): 8,503 admissions (5,342 train / 1,459 val / 1,702 test)
- Sepsis-restricted cohort (RL training/eval): 1,586 stays (1,036 train / 240 val / 310 test)
- State: transformer latents from Dynamic DeepHit (72h window)
- Actions: fluid + vasopressor dosing, normalized [0,1], heavily zero-inflated (fluid ~90.5% nonzero, vasopressor ~9.4% nonzero in test)
- Timesteps: hourly

**Planned next phases**: validation on independent London PICU datasets (harmonised already), possible external testing via collaboration with Prof. Kristen Gibbons (University of Queensland). Not yet started as of this summary.

---

## 1. Dynamic DeepHit — health score / reward model

Purpose: generate a dense, hourly "health score" to use as an RL reward signal, since paediatric mortality (<10%) is too low for a sparse survival-only reward (unlike the original adult AI Clinician, which had ~50% mortality).

- C-index: discharge=0.705, mortality=0.939, weighted=0.716; mortality AUROC=0.958
- `HealthScore = 50(1 + Σ p_{e,τ} w_{e,τ})`, rescaled to [0,100]
- Reward used for RL: `r_t = HealthScore_{t+1} − HealthScore_t`, discount γ=0.975
- An alternative **terminal reward** (Monte Carlo return, ±10 at episode end, discount γ=0.998, AI-Clinician-style) was built and rigorously compared against health-score — see Results below. Terminal reward was decisively rejected.

---

## 2. SARSA (`models/basic_q.py`, class `Q_Learner`)

Purpose: (a) policy-agnostic critic for advantage estimation of both real clinician behaviour and learned policies, (b) primary testbed for the health-score vs. terminal reward comparison, (c) later repurposed as the base for FQE (see §5).

- Architecture: twin Q-heads (Q1/Q2), LeakyReLU MLP, `Critic(state, action) -> (q1, q2)`
- Extensive **dose-response shape analysis** built around this critic: sweep each drug's dose across a grid holding the other fixed, classify the resulting Q(dose) curve as **increasing / decreasing / flat / non-monotonic**, and (for non-monotonic) fit a quadratic to further classify as **concave** (genuine interior optimum — clinically plausible) or **convex** (U-shaped — implausible).
- **Occupancy-support framework**: joint (severity × fluid-dose-bin × vaso-dose-bin) histograms built from training data, used throughout the project to check whether any given recommendation is backed by real precedent (`count ≥ min_count` at the matching bin). This became the recommended **clinical safety mechanism** (flag low-support recommendations to clinicians) rather than relying on model behaviour alone.

### Reward comparison result (SARSA, 5 seeds each)
| Metric | Terminal | Health-score |
|---|---|---|
| Vaso sensitivity (full cohort) | 7.6% | 45.9% |
| Fluid monotonicity | 97.9% (mostly convex/implausible when non-monotonic) | 96.6% (mostly concave/plausible when non-monotonic) |
| Q-outcome discrimination (AUC) | ~0.9 | ~0.51–0.55 (expected — near-term reward) |

Health-score reward was adopted as the basis for all subsequent RL work; terminal reward was carried forward only as a documented negative comparison.

---

## 3. CQL (`models/cql_with_logprobs.py`)

**Actor**: hurdle model — a Bernoulli **gate** (fire/no-fire) × Gaussian-in-logit **magnitude**, independently for fluid and vaso, sharing a common trunk.
**Critic**: standard CQL conservative penalty (logsumexp over random / current-policy / next-policy / vaso-only / zero-occupancy action samples).

### Major actor bugs found and fixed
1. **`f_std` floor collapse**: SAC-style policy-gradient loss (`-Q_π.mean()` on stochastic samples) creates an unconditional incentive to collapse variance toward zero when the critic's Q-surface is locally concave (Jensen's inequality). Fixed via SAC-style automatic entropy tuning: learnable dual variables (`log_alpha_mag_fluid/vaso`), target entropy derived from real-data raw-space std (fluid target_std=0.6721, vaso=0.6342).
2. **`f_raw_mean` saturation** (extreme positive tail, up to +20–35 in raw logit space): same underlying Q-gradient mechanism. Fixed via a smooth mean-saturation regularizer, weighted by gate uncertainty (`4·p·(1−p)`, active near p=0.5), scale ~0.05.

### Alpha (cql_alpha) selection
Extremely noisy at fine resolution (single-seed sweeps showed erratic, non-monotonic advantage across adjacent alpha values differing by 0.001 — later confirmed to be single-seed noise, not a real cliff). Final selection procedure (in order):
1. Exclude any alpha with own-critic advantage ≤ 0 (confirmed via stochastic-vs-deterministic re-evaluation to be a real training pathology at high alpha, not an eval artifact).
2. Exclude alphas at/above the point where SARSA-advantage becomes unstable (sliding-window ratio test).
3. Among survivors, select by **highest occupancy support** (safety-first), tie-break by SARSA-advantage.

Final 5 seeds trained at the selected fixed alpha.

### Residual known limitation (documented, not fully resolved)
A small subset of states (~2–4%) show a **near-maximal (~1.0) fluid dose recommendation**. Root-caused via the dose-response shape analysis: these are states with a genuinely **monotonically-increasing** Q(fluid dose) relationship (not a U-shaped/convex anomaly — convex states were shown to be *well*-supported by data). Of these monotonic-increasing states, only ~56–61% have real training-data support at the fluid=1 boundary (vs. ~100% support for concave/convex/decreasing categories) — i.e., "more fluid is better" conclusions are systematically the least well-grounded type of conclusion this critic reaches. **Recommended mitigation**: occupancy-based data-support flagging shown to clinicians per-recommendation, rather than further architecture tuning (explored and judged not worth the remaining effort given time constraints).

### Terminal-reward CQL: abandoned
Actor showed pathological bimodal (0-or-1) dosing for **both** fluid and vaso, at **every** tested alpha. Confirmed mechanistically via std analysis: `f_std`/`v_std` were saturated at either clamp boundary (floor or ceiling) in nearly 100% of states, regardless of alpha — the same architectural fixes that resolved health-score's actor did not generalise, consistent with terminal's poor/inconsistent underlying Q-landscape (near-flat action-sensitivity, convex-dominated non-monotonicity). No terminal-reward CQL model was carried forward.

---

## 4. BC (`models/bc_with_logprobs.py`) — behavioural cloning

Purpose: provide the **behaviour-policy density** for weighted importance sampling (WIS) — `log_prob_for_wis` in CQL borrows BC's fitted magnitude std as its bandwidth. Same hurdle-model actor architecture as CQL.

### History (important — informs what NOT to do)
- **Original baseline** (pure MLE loss, no entropy-tuning): clean, non-degenerate std for both drugs (0% at either clamp boundary). But had an **undiagnosed fluid gate miscalibration** (see below).
- **Mistake**: ported CQL's entropy-tuning dual-variable mechanism to BC "to mirror the actor." This **caused a regression**: vaso's std hit the ceiling in ~45–50% of states. Root cause: BC's pure-MLE objective never had CQL's Jensen's-inequality variance-collapse problem, so the entropy-tuning fix was solving a problem BC didn't have, while introducing a new one (a noisy dual-variable update, driven by vaso's sparse ~9% firing rate, that could overshoot toward the ceiling). **Entropy-tuning was removed entirely** for BC; the (independently safe) mean-saturation regularizer was kept, since it's weighted by broad gate-uncertainty, not masked to fired-only states, so it doesn't share the same sparse-signal fragility.
- **Fluid gate miscalibration** (present even in the clean original baseline): BC assigned only ~0.19–0.27 probability to the real "no fluid" decision at the ~9.5% of states where clinicians correctly withheld fluid (vs. vaso's well-calibrated ~0.94–0.98 in the analogous case). Quantified impact: this accounted for a median 33–39% (p90 73–82%) of each trajectory's total gate-related log-likelihood magnitude — large enough to seriously distort WIS importance weights.
- **Fix**: class-weighted BCE loss for the fluid gate specifically (upweight the rare "no fluid" class), applied to the training **loss only** — `log_prob_of_action`'s reported likelihood (used for WIS) is never reweighted, preserving correctness.
- **Tuning complication**: the effective strength of a given class-weight value depends on how well-converged training is — found only after a confusing round of confounded experiments. Learning rate also needed substantial reduction (original LR caused "all improvement in epoch 1, then val loss rises" — a convergence problem, not overtraining per se, ultimately requiring ~10x LR reduction from the original value).
- **Final validated configuration** (5 seeds): `fluid_gate_neg_class_weight=6.0`, no entropy-tuning, reduced LR. Results: fluid fire rate 70.6–71.8% (vs. true 90.5% — an accepted, explicit trade-off, see rationale below), WIS-impact fraction reduced to median 7.4–8.1% / p90 32.8–34.5% (a ~4–5x reduction from the uncorrected baseline). Vaso unaffected and remains well-calibrated throughout (~98% agreement at clinician-zero-vaso states, every seed).
- **Rationale for accepting the lower fire rate**: (1) BC's role is specifically to support WIS, not to match marginal clinical behaviour; (2) the residual under-firing is a much smaller, more benign distortion than the original near-universal firing (~96–98%), which reflected an essentially context-blind gate; (3) nonzero dose *magnitude* remains close to real values in both cases — only the fire/no-fire decision is affected; (4) the trade-off is now fully characterised and quantified, unlike the original unexamined miscalibration.
- **Explored but not adopted** (time-boxed, documented as future work): focal loss (single-seed test showed comparable/marginally-better tail behaviour, not validated across seeds); deeper fluid-gate MLP head (no measurable effect — ruled out gate capacity as the bottleneck); post-hoc Platt-scaling calibration (fails when stacked on an already class-weighted model — the two corrections compound and badly overshoot toward ~99%+ fire rate; would need a fair test against an *unweighted* baseline model, not built).

---

## 5. FQE — Fitted Q Evaluation (extends `models/basic_q.py`)

Algorithmically identical to SARSA, with **one change**: the Bellman target's `next_action` comes from a fixed, frozen **target policy** (the CQL actor) instead of the clinician's real logged next action. Implemented as a single conditional substitution (`Q_Learner._get_next_action`), gated by an optional `target_policy_actor` parameter — `None` reproduces exact SARSA behaviour.

- **Model loading**: `cql_with_logprobs.load_trained_actor(state_dim, action_dim, config_manager)` resolves which CQL actor to load based on `config_manager.search_variant`: `"random_search"` mode reads the *final chosen* `cql_alpha` from the `fixed_alpha` folder's own `config_manager.pkl`, then searches the separate `cql_alpha_sweep` folder for the matching run (needed because the original hyperparameter sweep predated the final alpha decision); `"best_hyperparameters"` mode matches by training seed within the `fixed_alpha` folder.
- **`training.py` fixes required**: added `"fqe"` to `MODEL_MODULES` (→ `basic_q`), and to the `model_type` whitelists in `copy_model_parameters`/`restore_model_parameters`/final checkpoint-saving (FQE's trainer is a `Q_Learner` with a `.critic`, not `.actor`/`.model` — these would otherwise crash at checkpoint time).
- **5 FQE models trained, one per CQL seed.** Critically, each FQE critic is only theoretically valid for evaluating the *one* target policy baked into its own training — pairing must be done by matching seed exactly, not by list order or a shared `config_manager.pkl` (which was found, in practice, to sometimes be shared across a whole seed sweep rather than per-run — the reliable source of truth is each model's own saved **checkpoint**, not `config_manager.pkl`).

### Result
Computed `Q_FQE(π_CQL(s), s) − Q_FQE(a_clinician, s)` for each of the 5 matched (CQL, FQE) seed pairs. All 5 positive (mean normalized advantage 0.0079, range 0.0038–0.0116). **Seed 4 showed the lowest advantage — the same seed independently flagged by SARSA earlier as anomalous** (degenerate vaso std, weakest SARSA-advantage). Two independent evaluation methods agreeing on the weakest seed is corroborating evidence that this is a genuine property of that trained policy, not an artifact of either evaluation method.

---

## Recurring methodological lessons (apply these going forward)

1. **Single-seed diagnostics are unreliable for fine-grained comparisons.** Repeatedly found across this project (CQL's fine alpha sweep, BC's LR/class-weight interaction) that adjacent single-seed results can differ by 5–10x purely from noise. Always validate a real conclusion across multiple seeds before trusting it.
2. **Verify seed/config matching directly from checkpoints**, never from a potentially-shared `config_manager.pkl` or assumed list order — this exact bug recurred independently for BC and for CQL/FQE pairing.
3. **Two-stage hyperparameter search**: use a single best-hyperparameter run (or a small sweep) to pick architecture/optimisation hyperparameters, then retrain 5 final seeds at the fixed, chosen configuration. Never skip straight to 5 seeds after a loss-function change without a single-seed sanity check first — every substantive loss change in this project surfaced at least one surprise on first run.
4. **Check computed vs. displayed precision separately** — e.g., Spearman ρ rounding to exactly 1.00 at 2 decimal places masked a real, small deviation from perfect monotonicity (true value ≈0.9985).
5. **A fix ported from one model to another isn't automatically safe** — CQL's entropy-tuning mechanism solved a real problem for CQL's stochastic policy-gradient objective, but actively harmed BC's unrelated MLE objective. Always ask whether the *mechanism* the fix targets is actually present before porting it.

---

## Key file map (`src/rl/`)
- `models/basic_q.py` — SARSA critic; also the FQE trainer (`Q_Learner` with optional `target_policy_actor`)
- `models/cql_with_logprobs.py` — CQL actor + critic; `load_trained_actor` for FQE's target-policy loading
- `models/bc_with_logprobs.py` — BC actor (class-weighted fluid gate, mean-saturation regularizer, no entropy-tuning; also has unused/exploratory `focal` loss, `fluid_gate_hidden`, and `calibrate_fluid_gate` options)
- `training.py` — shared training loop, `MODEL_MODULES`, checkpoint saving
- `dose_response_shapes.py` — dose-response sweep, shape classification, occupancy-support machinery
- `evaluate_cql_alpha_sweep.py`, `evaluate_terminal_reward_models.py`, `evaluate_bc_final_seeds.py`, `evaluate_fqe_final_seeds.py` — per-model analysis scripts
- `action_distribution_sweep.py`, `dose_histogram_by_alpha.py`, `clinician_baseline.py`, `check_std_floor.py`, `shape_dose_overlap.py` — supporting plotting/diagnostic utilities

## Not yet done (as of this summary)
- Weighted importance sampling (WIS) using the validated BC models — about to begin
- London PICU dataset validation
- External testing (Queensland collaboration)
- Manuscript(s): one on the reward-function comparison + internal validation methodology is in preparation
