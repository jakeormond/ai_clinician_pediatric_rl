"""
Generate Time-Dependent Cause-Specific Concordance Index Table

This script creates a table showing C_k(t, Δt) - the cause-specific time-dependent
concordance index that measures:
"Among patients still alive at time t, how well can the model rank who will
experience the event within the next Δt hours?"

Table Structure:
- Columns: 5 Δt bins × 2 events (death, discharge) = 10 columns
- Rows: 7 models × 3 t-groups = 21 rows

Each cell shows: C-index (± std) with * for p < 0.05 vs baseline
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config_manager import ConfigManager
from src.data.data_processor_for_shards import ShardProcessor
from src.supervised_learning.training import create_processor_from_shards
from src.supervised_learning.models.dynamic_deephit_transformer import TransformerDeepHitMulti


# ============================================================================
# Configuration
# ============================================================================

# Time bins from the model (in hours)
TIME_BINS = [0, 13, 28, 47, 75, 116, 185, 322, 1435]

# Δt columns to include (indices into TIME_BINS)
DELTA_T_BINS = {
    0: "0-13h",      # bin 0
    2: "28-47h",     # bin 2
    3: "47-75h",     # bin 3
    5: "116-185h",   # bin 5
    7: "322-1435h",  # bin 7
}

# t groups (ICU stay duration at prediction time, in hours)
T_GROUPS = {
    "short": {"label": "t < 48h (short ICU stays)", "min": 0, "max": 48},
    "medium": {"label": "t = 48-72h (medium ICU stays)", "min": 48, "max": 72},
    "long": {"label": "t > 72h (long ICU stays)", "min": 72, "max": float('inf')},
}

# Event types
EVENT_TYPES = {
    0: "Discharge",  # Usually event 0 is discharge/survival
    1: "Death",      # Usually event 1 is death
}

# Cohorts
COHORTS = [
    "all",
    "sepsis_broad",
    "sepsis_strict",
    "ards_broad",
    "ards_strict",
    "aki_broad",
    "aki_strict",
]


# ============================================================================
# C-index Calculation Functions
# ============================================================================

def calculate_cif(logits: torch.Tensor) -> torch.Tensor:
    """Calculate Cumulative Incidence Function from logits."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    logits = logits.contiguous()
    B, K, T = logits.shape
    logits_flat = logits.view(B, K * T)
    p = F.softmax(logits_flat, dim=1).view(B, K, T)
    cif = p.cumsum(dim=2)
    return cif


def calculate_c_index_for_bin(
    logits: torch.Tensor,
    time_to_event: torch.Tensor,
    event_label: torch.Tensor,
    event_k: int,
    time_bin_t: int,
    max_samples: int = 10000
) -> float:
    """
    Calculate C-index for a specific event type (k) and time bin (t).

    This computes C_k(t) which measures the concordance for cause k
    at time horizon t.
    """
    B, K, T = logits.shape

    # Subsample if needed
    if B > max_samples:
        perm = torch.randperm(B)[:max_samples]
        logits = logits[perm]
        time_to_event = time_to_event[perm]
        event_label = event_label[perm]
        B = max_samples

    with torch.inference_mode():
        cif = calculate_cif(logits)

        # Get CIF scores for this event type and time bin
        s_t = cif[:, event_k, time_bin_t].cpu()  # [B]
        tt = time_to_event.cpu()
        ev = event_label.cpu()

        # Cases: patients with event k and event time <= t
        cases_mask = (ev == event_k) & (tt <= time_bin_t)
        idx_cases = torch.nonzero(cases_mask, as_tuple=False).squeeze(1)
        num_cases = idx_cases.numel()

        if num_cases == 0:
            return float('nan')

        # Calculate concordance
        conc_total = 0.0
        comp_total = 0.0
        chunk_size = 2048

        tt_all = tt.unsqueeze(0)
        ev_all = ev.unsqueeze(0)
        s_all = s_t.unsqueeze(0)

        for start in range(0, num_cases, chunk_size):
            sel = idx_cases[start:start+chunk_size]
            tt_i = tt[sel].unsqueeze(1)
            s_i = s_t[sel].unsqueeze(1)

            # Valid comparators: (time_j >= time_i) OR (event_j != k)
            valid_j = (tt_all >= tt_i) | (ev_all != event_k)

            # Concordant: s_i > s_j among valid_j
            conc = ((s_i > s_all) & valid_j).sum(dim=1)
            comp = valid_j.sum(dim=1)

            conc_total += float(conc.sum())
            comp_total += float(comp.sum())

        if comp_total > 0:
            return conc_total / comp_total
        else:
            return float('nan')


def bootstrap_c_index(
    logits: torch.Tensor,
    time_to_event: torch.Tensor,
    event_label: torch.Tensor,
    event_k: int,
    time_bin_t: int,
    n_bootstrap: int = 100,
    sample_frac: float = 0.8
) -> Tuple[float, float, List[float]]:
    """
    Bootstrap C-index calculation to get mean, std, and all samples.
    """
    B = logits.shape[0]
    sample_size = int(B * sample_frac)

    c_indices = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = torch.randint(0, B, (sample_size,))

        c_idx = calculate_c_index_for_bin(
            logits[indices],
            time_to_event[indices],
            event_label[indices],
            event_k,
            time_bin_t,
            max_samples=5000
        )

        if not np.isnan(c_idx):
            c_indices.append(c_idx)

    if len(c_indices) == 0:
        return float('nan'), float('nan'), []

    return np.mean(c_indices), np.std(c_indices), c_indices


# ============================================================================
# Data Loading Functions
# ============================================================================

def find_latest_experiment(experiments_dir: str, cohort: str) -> Optional[str]:
    """Find the latest experiment directory for a given cohort."""
    if cohort == 'all':
        pattern = "mimic_iv_dynamic_deephit_transformer_discharge_mortality_tte_standard_*"
    else:
        pattern = f"mimic_iv_dynamic_deephit_transformer_discharge_mortality_tte_standard_{cohort}_*"

    matching_dirs = glob.glob(os.path.join(experiments_dir, pattern))

    # Filter for 'all' cohort
    if cohort == 'all':
        cohort_suffixes = ['_aki_broad', '_aki_strict', '_ards_broad', '_ards_strict',
                          '_sepsis_broad', '_sepsis_strict']
        matching_dirs = [d for d in matching_dirs
                        if not any(suffix in os.path.basename(d) for suffix in cohort_suffixes)]

    # Filter to only directories that contain final_model.pt
    matching_dirs = [d for d in matching_dirs
                    if os.path.exists(os.path.join(d, 'final_model.pt'))]

    if not matching_dirs:
        return None

    matching_dirs.sort(key=os.path.getmtime, reverse=True)
    return matching_dirs[0]


def load_predictions(
    experiment_dir: str,
    config_manager: 'ConfigManager' = None,
    generate_if_missing: bool = True,
    device: str = 'cuda'
) -> Optional[Dict]:
    """
    Load predictions from an experiment directory.
    If predictions don't exist and generate_if_missing=True, generate them.
    """
    filepath = os.path.join(experiment_dir, 'test_set_predictions.h5')

    if os.path.exists(filepath):
        print(f"Loading existing predictions from {filepath}")
        return ShardProcessor.load_dynamic_deephit_dataset_from_h5(filepath)

    if not generate_if_missing or config_manager is None:
        print(f"WARNING: No predictions found at {filepath}")
        return None

    print(f"Generating predictions for {experiment_dir}...")

    # Load results.json to get hyperparameters
    results_path = os.path.join(experiment_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f"WARNING: No results.json found at {results_path}")
        return None

    with open(results_path, 'r') as f:
        results_dict = json.load(f)

    hyperparams = results_dict['best_hyperparameters']
    time_bins = results_dict['time_bins']
    n_time_bins = len(time_bins) - 1

    # Create test processor
    _, _, test_processor = create_processor_from_shards(config_manager, normalize=True)
    feature_dim = test_processor.feature_dim

    # Instantiate model
    # Note: 'n_layers' in hyperparams is the actual encoder layer count used during training
    # 'num_encoder_layers' in the JSON may be incorrect, so we use 'n_layers' as fallback
    num_encoder_layers = hyperparams.get('n_layers', hyperparams.get('num_encoder_layers', 2))
    window_length = hyperparams.get('window_length', 72)

    model = TransformerDeepHitMulti(
        feature_dim=feature_dim,
        d_model=hyperparams.get('d_model', 128),
        nhead=hyperparams.get('nhead', 8),
        num_encoder_layers=num_encoder_layers,
        ff_factor=hyperparams.get('ff_factor', 2),
        num_time_bins=n_time_bins,
        num_risks=2,
        dropout=hyperparams.get('dropout', 0.2),
        max_seq_len=window_length,
        predict_covariates=test_processor.physiological_feature_ind,
    ).to(device)

    # Load model weights
    model_path = os.path.join(experiment_dir, 'final_model.pt')
    if not os.path.exists(model_path):
        print(f"WARNING: No model found at {model_path}")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run inference
    dataset = run_inference(model, test_processor, config_manager, experiment_dir, device)

    return dataset


def run_inference(
    model: torch.nn.Module,
    processor: 'ShardProcessor',
    config_manager: 'ConfigManager',
    experiment_dir: str,
    device: str = 'cuda'
) -> Dict:
    """Run inference on test set and save predictions."""
    stay_ids = processor.dynamic_deephit_dataset['stay_ids']
    index_ptr = processor.dynamic_deephit_dataset['index_ptr']
    window_length = config_manager.training_config['window_length']

    all_predictions = []
    model.eval()

    print(f"Running inference on {len(stay_ids)} patients...")

    with torch.no_grad():
        for i, stay_id in enumerate(tqdm(stay_ids, desc="Inference")):
            start_idx = index_ptr[i]
            end_idx = index_ptr[i + 1]

            trajectory = processor.dynamic_deephit_dataset['trajectories'][start_idx:end_idx]
            trajectory = torch.as_tensor(trajectory, dtype=torch.float32, device=device)
            T, F = trajectory.shape

            # Add padding
            pad = trajectory[0:1].repeat(window_length - 1, 1)
            padded_traj = torch.cat([pad, trajectory], dim=0)

            # Create windows
            windows = padded_traj.unfold(0, window_length, 1).transpose(1, 2)

            # Batch inference
            batch_size = 64
            all_preds = []

            for chunk_start in range(0, T, batch_size):
                chunk_end = min(chunk_start + batch_size, T)
                windows_chunk = windows[chunk_start:chunk_end].to(device)

                risk_logits, _ = model(windows_chunk)
                risk_logits = [logits[:, -1, :].cpu() for logits in risk_logits]
                risk_logits = torch.stack(risk_logits, dim=2)
                final_step_logits = risk_logits.permute(0, 2, 1)
                all_preds.append(final_step_logits)

            patient_preds = torch.cat(all_preds, dim=0)
            all_predictions.append(patient_preds)

    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    processor.dynamic_deephit_dataset['predictions'] = all_predictions

    # Save to h5 file
    file_path = os.path.join(experiment_dir, 'test_set_predictions.h5')
    processor.dynamic_deephit_dataset_to_h5(file_path)
    print(f"Saved predictions to {file_path}")

    return processor.dynamic_deephit_dataset


def load_cohort_stay_ids(cohorts_dir: str, cohort: str) -> set:
    """Load stay IDs for a specific cohort."""
    if cohort == 'all':
        return None  # No filtering needed

    stay_ids_path = os.path.join(cohorts_dir, cohort, 'stay_ids.csv')
    if os.path.exists(stay_ids_path):
        df = pd.read_csv(stay_ids_path)
        return set(df['stay_id'].unique())
    else:
        print(f"WARNING: Cohort file not found: {stay_ids_path}")
        return None


def filter_by_cohort(dataset: Dict, cohort_stay_ids: set) -> Dict:
    """Filter dataset to only include stays in the cohort."""
    if cohort_stay_ids is None:
        return dataset

    stay_ids = dataset['stay_ids']
    mask = np.isin(stay_ids, list(cohort_stay_ids))
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    # Build trajectory mask
    traj_mask = np.zeros(dataset['trajectories'].shape[0], dtype=bool)
    for idx in indices:
        start = dataset['index_ptr'][idx]
        end = dataset['index_ptr'][idx + 1]
        traj_mask[start:end] = True

    # Create filtered dataset
    filtered = {
        'stay_ids': dataset['stay_ids'][mask],
        'event': dataset['event'][mask],
        'predictions': dataset['predictions'][traj_mask],
        'time_to_event': dataset['time_to_event'][traj_mask],
        'time_to_event_raw': dataset['time_to_event_raw'][traj_mask],
        'trajectories': dataset['trajectories'][traj_mask],
    }

    # Rebuild index_ptr
    new_index_ptr = np.zeros(len(indices) + 1, dtype=np.int32)
    for i, idx in enumerate(indices):
        start = dataset['index_ptr'][idx]
        end = dataset['index_ptr'][idx + 1]
        new_index_ptr[i + 1] = new_index_ptr[i] + (end - start)
    filtered['index_ptr'] = new_index_ptr

    return filtered


def filter_by_time_in_icu(
    dataset: Dict,
    t_min: int,
    t_max: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter predictions to only include those made at time t in ICU.

    Returns:
        logits: [N, K, T] predictions
        time_to_event: [N] time bins until event
        event_label: [N] event type (expanded to per-timestep)
    """
    predictions = dataset['predictions']
    time_to_event = dataset['time_to_event']
    events = dataset['event']
    index_ptr = dataset['index_ptr']

    # Create per-timestep event labels
    n_timesteps = predictions.shape[0]
    event_labels = np.zeros(n_timesteps, dtype=np.int32)
    timestep_in_icu = np.zeros(n_timesteps, dtype=np.int32)

    for i in range(len(events)):
        start = index_ptr[i]
        end = index_ptr[i + 1]
        event_labels[start:end] = events[i]
        timestep_in_icu[start:end] = np.arange(end - start)

    # Filter by time in ICU
    mask = (timestep_in_icu >= t_min) & (timestep_in_icu < t_max)

    if mask.sum() == 0:
        return None, None, None

    logits = torch.from_numpy(predictions[mask])
    tte = torch.from_numpy(time_to_event[mask])
    ev = torch.from_numpy(event_labels[mask])

    return logits, tte, ev


# ============================================================================
# Statistical Tests
# ============================================================================

def paired_permutation_test(
    samples_baseline: List[float],
    samples_cohort: List[float],
    n_permutations: int = 1000
) -> float:
    """
    Perform paired permutation test to compare two sets of bootstrap samples.
    Tests if cohort model is significantly different from baseline.
    """
    if len(samples_baseline) == 0 or len(samples_cohort) == 0:
        return 1.0

    # Use minimum length
    n = min(len(samples_baseline), len(samples_cohort))
    baseline = np.array(samples_baseline[:n])
    cohort = np.array(samples_cohort[:n])

    # Observed difference
    obs_diff = np.mean(cohort) - np.mean(baseline)

    # Permutation test
    combined = np.stack([baseline, cohort], axis=1)
    count_extreme = 0

    for _ in range(n_permutations):
        # Randomly swap pairs
        swaps = np.random.randint(0, 2, n)
        perm_baseline = np.where(swaps == 0, combined[:, 0], combined[:, 1])
        perm_cohort = np.where(swaps == 0, combined[:, 1], combined[:, 0])

        perm_diff = np.mean(perm_cohort) - np.mean(perm_baseline)

        if abs(perm_diff) >= abs(obs_diff):
            count_extreme += 1

    return count_extreme / n_permutations


# ============================================================================
# Table Generation
# ============================================================================

def generate_table(
    results: Dict,
    baseline_results: Dict,
    cohort_sizes: Dict[str, int],
    total_size: int
) -> pd.DataFrame:
    """
    Generate the final table as a pandas DataFrame.

    Args:
        results: Dict of (cohort, t_group, delta_t, event_k) -> (mean, std, samples)
                 Contains cohort model results on cohort patients
        baseline_results: Dict of (cohort, t_group, delta_t, event_k) -> (mean, std, samples)
                          Contains 'all' model results on cohort patients (for fair comparison)
        cohort_sizes: Dict of cohort -> number of TEST patients in that cohort
        total_size: Total number of test patients
    """
    # Build column names
    columns = ['Model/Cohort', 't Group']
    for delta_t_idx, delta_t_label in DELTA_T_BINS.items():
        for event_type in ['Death', 'Discharge']:
            columns.append(f"{event_type}\nΔt={delta_t_label}")

    rows = []

    # Hardcoded percentages based on training data predictions, specific to each prediction time group
    # These represent % of "All" predictions made at each time point (not total LOS)
    # Format: COHORT_PERCENTAGES[t_group_key][cohort] = percentage
    # Based on 5,742,498 total predictions across 53,017 training stays
    COHORT_PERCENTAGES = {
        'short': {  # Predictions made in first 48 hours (t < 48h)
            'all': '100%',
            'sepsis_broad': '47.5%',
            'sepsis_strict': '20.1%',
            'aki_broad': '52.4%',
            'aki_strict': '17.8%',
            'ards_broad': '23.0%',
            'ards_strict': '12.4%',
        },
        'medium': {  # Predictions made between 48-72 hours (t = 48-72h)
            'all': '100%',
            'sepsis_broad': '58.2%',
            'sepsis_strict': '25.9%',
            'aki_broad': '65.2%',
            'aki_strict': '23.7%',
            'ards_broad': '33.5%',
            'ards_strict': '19.6%',
        },
        'long': {  # Predictions made after 72 hours (t > 72h)
            'all': '100%',
            'sepsis_broad': '71.5%',
            'sepsis_strict': '38.5%',
            'aki_broad': '74.9%',
            'aki_strict': '32.0%',
            'ards_broad': '54.0%',
            'ards_strict': '37.8%',
        },
    }

    for t_group_key, t_group_info in T_GROUPS.items():
        for cohort in COHORTS:
            if cohort == 'all':
                cohort_label = f"All (100%) (Baseline)"
            else:
                pct = COHORT_PERCENTAGES.get(t_group_key, {}).get(cohort, 'N/A')
                cohort_label = f"{cohort.replace('_', ' ').title()} ({pct})"

            row = [cohort_label, t_group_info['label']]

            for delta_t_idx in DELTA_T_BINS.keys():
                for event_k, event_name in EVENT_TYPES.items():
                    key = (cohort, t_group_key, delta_t_idx, event_k)

                    if key in results:
                        mean, std, samples = results[key]

                        # Check significance vs baseline
                        # For cohort rows, compare against 'all' model on SAME cohort patients
                        # (stored in baseline_results), not 'all' model on ALL patients
                        sig_marker = ""
                        if cohort != 'all' and key in baseline_results:
                            _, _, baseline_samples = baseline_results[key]
                            p_value = paired_permutation_test(baseline_samples, samples)
                            if p_value < 0.001:
                                sig_marker = "***"
                            elif p_value < 0.01:
                                sig_marker = "**"
                            elif p_value < 0.05:
                                sig_marker = "*"

                        if np.isnan(mean):
                            cell = "N/A"
                        else:
                            cell = f"{mean:.3f} (±{std:.3f}){sig_marker}"
                    else:
                        cell = "N/A"

                    row.append(cell)

            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    return df


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Time-Dependent C-index Table'
    )
    parser.add_argument('--hospital', type=str, default='mimic_iv')
    parser.add_argument('--n_bootstrap', type=int, default=100,
                       help='Number of bootstrap samples')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--cohorts', type=str, nargs='+', default=COHORTS,
                       help='Cohorts to include')

    args = parser.parse_args()

    # Initialize config manager
    # ConfigManager expects an args namespace object
    from argparse import Namespace
    config_args = Namespace(
        hospital=args.hospital,
        experiment_type='supervised_learning',
        model_type='dynamic_deephit_transformer',
        prediction_type='discharge_mortality_tte',
        mode='single_split',
        search_variant='fixed_hyperparameters',
        experiment_variant='standard',
        cohort='all',
        n_epochs=1,
        start_time='evaluation',
        final_timestep_only=False,
        n_search_iterations=1,
        action_mode='continuous_only'
    )
    config_manager = ConfigManager(config_args)
    config_manager.set_training_config()

    experiments_dir = config_manager.paths['experiments_dir']
    cohorts_dir = os.path.join(config_manager.paths['data_dir'], 'cohorts')
    output_dir = args.output_dir or config_manager.paths['final_model_dir']

    print("=" * 80)
    print("Time-Dependent Cause-Specific C-index Table Generator")
    print("=" * 80)
    print(f"Experiments directory: {experiments_dir}")
    print(f"Cohorts directory: {cohorts_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Cohorts: {args.cohorts}")
    print("=" * 80)

    # Load predictions for 'all' model first (baseline)
    print("\n[Step 1] Loading baseline model predictions (all data)...")
    all_exp_dir = find_latest_experiment(experiments_dir, 'all')
    if all_exp_dir is None:
        print("ERROR: No experiment found for 'all' cohort. Exiting.")
        sys.exit(1)

    all_predictions = load_predictions(all_exp_dir, config_manager, generate_if_missing=True)
    if all_predictions is None:
        print("ERROR: Could not load predictions for 'all' model. Exiting.")
        sys.exit(1)

    total_size = len(all_predictions['stay_ids'])
    all_test_stay_ids = set(all_predictions['stay_ids'])
    print(f"Loaded {total_size} patients from baseline model.")

    # Load cohort stay IDs and count sizes
    # IMPORTANT: cohort_sizes should be the number of TEST patients in each cohort,
    # not the total from the CSV (which includes train+val+test)
    cohort_sizes = {'all': total_size}
    cohort_stay_ids = {'all': None}

    for cohort in args.cohorts:
        if cohort != 'all':
            stay_ids = load_cohort_stay_ids(cohorts_dir, cohort)
            if stay_ids is not None:
                cohort_stay_ids[cohort] = stay_ids
                # Calculate size as intersection of test set and cohort (not total CSV count)
                test_cohort_intersection = all_test_stay_ids & stay_ids
                cohort_sizes[cohort] = len(test_cohort_intersection)
                print(f"Loaded {len(stay_ids)} stay IDs for cohort '{cohort}' ({cohort_sizes[cohort]} in test set)")

    # Results storage
    results = {}

    # Process each cohort
    print("\n[Step 2] Computing C-indices...")

    # For fair comparison, we need to evaluate both 'all' model and cohort model
    # on the SAME patients (the cohort patients in the test set)
    # Store baseline ('all' model) results for each cohort separately
    baseline_results = {}  # key: (cohort, t_group, delta_t, event_k) -> results for 'all' model on that cohort

    for cohort in tqdm(args.cohorts, desc="Cohorts"):
        # Get the cohort patient subset from the 'all' predictions
        if cohort == 'all':
            # For 'all' row, use all test patients
            dataset = all_predictions
            cohort_filter_ids = None
        else:
            # For cohort rows, filter 'all' predictions to cohort patients
            if cohort not in cohort_stay_ids or cohort_stay_ids[cohort] is None:
                print(f"WARNING: No cohort stay IDs for '{cohort}'. Skipping.")
                continue

            cohort_filter_ids = cohort_stay_ids[cohort]

            # First, compute baseline ('all' model on cohort patients) for significance testing
            baseline_dataset = filter_by_cohort(all_predictions, cohort_filter_ids)
            if baseline_dataset is None:
                print(f"WARNING: No matching stays in 'all' predictions for cohort '{cohort}'. Skipping.")
                continue

            # Load cohort-specific model predictions
            # IMPORTANT: Don't call reset_cohort - we want to use the 'all' test set
            exp_dir = find_latest_experiment(experiments_dir, cohort)
            if exp_dir is None:
                print(f"WARNING: No experiment found for cohort '{cohort}'. Skipping.")
                continue

            # Load cohort model predictions (may need to regenerate on 'all' test data)
            # Keep config_manager on 'all' cohort so it uses the full test set
            config_manager.reset_cohort('all')
            dataset = load_predictions(exp_dir, config_manager, generate_if_missing=True)
            if dataset is None:
                print(f"WARNING: Could not load predictions for cohort '{cohort}'. Skipping.")
                continue

            # Filter cohort model predictions to cohort patients
            dataset = filter_by_cohort(dataset, cohort_filter_ids)
            if dataset is None:
                print(f"WARNING: No matching stays for cohort '{cohort}'. Skipping.")
                continue

            # Compute and store baseline results for this cohort (for significance testing)
            for t_group_key, t_group_info in T_GROUPS.items():
                t_min = t_group_info['min']
                t_max = t_group_info['max']
                logits, tte, ev = filter_by_time_in_icu(baseline_dataset, t_min, t_max)
                if logits is None or logits.shape[0] < 100:
                    continue
                for delta_t_idx in DELTA_T_BINS.keys():
                    for event_k in EVENT_TYPES.keys():
                        mean, std, samples = bootstrap_c_index(
                            logits, tte, ev, event_k=event_k,
                            time_bin_t=delta_t_idx, n_bootstrap=args.n_bootstrap
                        )
                        baseline_key = (cohort, t_group_key, delta_t_idx, event_k)
                        baseline_results[baseline_key] = (mean, std, samples)

        # Process each t-group
        for t_group_key, t_group_info in T_GROUPS.items():
            t_min = t_group_info['min']
            t_max = t_group_info['max']

            # Filter by time in ICU
            logits, tte, ev = filter_by_time_in_icu(dataset, t_min, t_max)

            if logits is None or logits.shape[0] < 100:
                print(f"  Skipping {cohort}/{t_group_key}: insufficient samples")
                continue

            # Calculate C-index for each Δt bin and event type
            for delta_t_idx in DELTA_T_BINS.keys():
                for event_k in EVENT_TYPES.keys():
                    mean, std, samples = bootstrap_c_index(
                        logits, tte, ev,
                        event_k=event_k,
                        time_bin_t=delta_t_idx,
                        n_bootstrap=args.n_bootstrap
                    )

                    key = (cohort, t_group_key, delta_t_idx, event_k)
                    results[key] = (mean, std, samples)

    # Generate table
    print("\n[Step 3] Generating table...")
    df = generate_table(results, baseline_results, cohort_sizes, total_size)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'time_dependent_cindex_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV table to: {csv_path}")

    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'time_dependent_cindex_table.tex')
    latex_str = df.to_latex(index=False, escape=False, column_format='ll' + 'c' * 10)
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to: {latex_path}")

    # Save raw results as JSON
    json_results = {}
    for key, (mean, std, samples) in results.items():
        cohort, t_group, delta_t, event_k = key
        json_key = f"{cohort}_{t_group}_{delta_t}_{event_k}"
        json_results[json_key] = {
            'mean': float(mean) if not np.isnan(mean) else None,
            'std': float(std) if not np.isnan(std) else None,
            'n_samples': len(samples)
        }

    json_path = os.path.join(output_dir, 'time_dependent_cindex_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved raw results to: {json_path}")

    # Print table
    print("\n" + "=" * 80)
    print("TIME-DEPENDENT C-INDEX TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001 vs baseline (all)")
    print("=" * 80)


if __name__ == "__main__":
    main()
