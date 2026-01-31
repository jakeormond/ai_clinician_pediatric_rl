import numpy as np
import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, \
    average_precision_score, roc_curve, precision_recall_curve

from typing import Dict


import logging
logger = logging.getLogger(__name__)


def calculate_probabilities(
    logits: torch.Tensor
    ):
    # Ensure logits is a torch tensor
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    
    # Ensure tensor is contiguous before view
    logits = logits.contiguous()
    B, K, T = logits.shape
    logits_flat = logits.view(B, K * T)   
    p = F.softmax(logits_flat, dim=1).view(B, K, T)   # [B, K, T]

    return p


def calculate_cif(
    logits: torch.Tensor
    ):
    
    # Ensure logits is a torch tensor
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    
    # Ensure tensor is contiguous before view
    logits = logits.contiguous()

    B, K, T = logits.shape
    logits_flat = logits.view(B, K * T)   
    p = F.softmax(logits_flat, dim=1).view(B, K, T)   # [B, K, T]

    # 2) CIF: cumulative over time
    cif = p.cumsum(dim=2)  

    return cif

def _choose_chunk_size(B, target_gb=2.0, cap=16384):
    avail = psutil.virtual_memory().available
    # Use at most target_gb or 50% of available, whichever smaller
    budget = min(target_gb * (1024**3), avail * 0.5)
    max_chunk = int(budget / B)  # bytes per bool
    return max(512, min(max_chunk, cap))

def calculate_c_index(logits, time_to_event, event_label, random_sample=True) -> Dict[str, float]:
    B, K, T = logits.shape
    if random_sample and B > 10000:
        # randomly sample 10k for C-index calculation to limit memory use
        perm = torch.randperm(B)
        sel = perm[:10000]
        logits = logits[sel]
        time_to_event = time_to_event[sel]
        event_label = event_label[sel]
        B = 10000
        logger.info(f"Calculating C-index on random sample of 10,000 from {B} to limit memory usage.")

    elif B > 10000:
        # for validation/test, keep it deterministic: take first 10k
        logits = logits[:10000]
        time_to_event = time_to_event[:10000]
        event_label = event_label[:10000]
        B = 10000
        logger.info(f"Calculating C-index on first 10,000 from {B} to limit memory usage.")

    with torch.inference_mode():
        cif = calculate_cif(logits)
        tt = time_to_event.cpu()
        ev = event_label.cpu()
        cif_cpu = cif.cpu()
        C = torch.empty(K, T)
        base_chunk = _choose_chunk_size(B, target_gb=2.0, cap=16384)
        for k in range(K):
            for t in range(T):
                s_t = cif_cpu[:, k, t]            # [B]
                cases_mask = (ev == k) & (tt <= t)
                idx_cases = torch.nonzero(cases_mask, as_tuple=False).squeeze(1)
                num_cases = idx_cases.numel()
                if num_cases == 0:
                    C[k, t] = torch.nan
                    continue
                chunk_size = base_chunk
                conc_total = 0.0
                comp_total = 0.0
                # Precompute broadcast tensors once
                tt_all = tt.unsqueeze(0)          # [1,B]
                ev_all = ev.unsqueeze(0)          # [1,B]
                s_all = s_t.unsqueeze(0)          # [1,B]
                for start in range(0, num_cases, chunk_size):
                    sel = idx_cases[start:start+chunk_size]
                    tt_i = tt[sel].unsqueeze(1)   # [C,1]
                    s_i = s_t[sel].unsqueeze(1)   # [C,1]
                    # valid_j: (time_j >= time_i) OR (event_j != k)
                    valid_j = (tt_all >= tt_i) | (ev_all != k)         # [C,B]
                    # concordant: s_i > s_j among valid_j
                    conc = ((s_i > s_all) & valid_j).sum(dim=1)
                    comp = valid_j.sum(dim=1)
                    conc_total += float(conc.sum())
                    comp_total += float(comp.sum())
                C[k, t] = conc_total / comp_total if comp_total > 0 else torch.nan
        C_integrated_k = torch.nanmean(C, dim=1)
        # Weighted aggregation
        true_time_onehot = torch.zeros(B, T)
        true_time_onehot.scatter_(1, tt.long().view(B,1), 1.0)
        true_cause_onehot = torch.zeros(B, K)
        true_cause_onehot.scatter_(1, ev.view(B,1).long(), 1.0)
        event_occ = true_cause_onehot.unsqueeze(2) * true_time_onehot.unsqueeze(1)  # [B,K,T]
        E_k_t = event_occ.sum(dim=0)              # [K,T]
        denom_k = E_k_t.sum(dim=1)
        denom_k_safe = torch.where(denom_k == 0, torch.tensor(1.0), denom_k)
        C_weighted_k = (torch.nan_to_num(C) * E_k_t).sum(dim=1) / denom_k_safe
        denom_all = denom_k_safe.sum()
        C_weighted_multi = (C_weighted_k * denom_k_safe).sum() / denom_all
        return {
            "C_k_t": C,
            "C_integrated_k": C_integrated_k,
            "C_weighted_k": C_weighted_k,
            "C_weighted_multi": C_weighted_multi.item()
        }

# def calculate_c_index(
#     logits: torch.Tensor,
#     time_to_event: torch.Tensor,
#     event_label: torch.Tensor,
# ) -> Dict[str, float]:
    
#     # Compute CIF first
#     B, K, T = logits.shape
#     device = logits.device

#     with torch.no_grad():
#         cif = calculate_cif(logits)  # [B, K, T]
#         C = torch.empty(K, T, device='cpu')  # store on CPU to limit GPU mem

#         # Precompute on CPU to reduce memory pressure
#         tt = time_to_event.to('cpu')
#         ev = event_label.to('cpu')
#         cif_cpu = cif.to('cpu')

#         # Tunable chunk size to bound memory usage: memory ~ chunk_size * B
#         # For B=100k and chunk=512 => ~50M bools (~50–100MB)
#         chunk_size = 16384

#         for k in range(K):
#             for t in range(T):
#                 # Cases: patients with event k and event time <= t
#                 cases_mask = (ev == k) & (tt <= t)
#                 num_cases = int(cases_mask.sum())
#                 if num_cases == 0:
#                     C[k, t] = torch.nan
#                     continue

#                 s_t = cif_cpu[:, k, t]  # [B]
#                 idx_cases = torch.nonzero(cases_mask, as_tuple=False).squeeze(1)

#                 conc_total = 0.0
#                 comp_total = 0.0

#                 for start in range(0, num_cases, chunk_size):
#                     idx_i = idx_cases[start:start + chunk_size]            # [C]
#                     tt_i = tt[idx_i].unsqueeze(1)                          # [C,1]
#                     s_i = s_t[idx_i].unsqueeze(1)                          # [C,1]

#                     # Valid comparators j for each i: (time_j >= time_i) OR (event_j != k)
#                     # Shape [C,B], computed in chunks to control memory
#                     valid_j = (tt.unsqueeze(0) >= tt_i) | (ev.unsqueeze(0) != k)  # [C,B]

#                     # Concordant: s_i > s_j among valid_j
#                     # Broadcast s_t -> [1,B], s_i -> [C,1]
#                     conc = ((s_i > s_t.unsqueeze(0)) & valid_j).sum(dim=1).float()  # [C]
#                     comp = valid_j.sum(dim=1).float()                                # [C]

#                     conc_total += float(conc.sum())
#                     comp_total += float(comp.sum())

#                 if comp_total > 0:
#                     C[k, t] = conc_total / comp_total
#                 else:
#                     C[k, t] = torch.nan

#         # integrated cause-specific C-index
#         C_integrated_k = torch.nanmean(C, dim=1)  # [K]

#         # event-weighted
#         true_time_onehot = torch.zeros(B, T)
#         true_time_onehot.scatter_(1, tt.long().view(B, 1), 1.0)  # [B,T]

#         true_cause_onehot = torch.zeros(B, K)
#         true_cause_onehot.scatter_(1, ev.view(B, 1).long(), 1.0)  # [B, K]

#         event_occ = true_cause_onehot.unsqueeze(2) * true_time_onehot.unsqueeze(1)  # [B,K,T]
#         E_k_t = event_occ.sum(dim=0)  # [K, T]
#         denom_k = E_k_t.sum(dim=1)    # [K]
#         # Guard against zero division
#         denom_k = torch.where(denom_k == 0, torch.tensor(1.0), denom_k)
#         C_weighted_k = (torch.nan_to_num(C) * E_k_t).sum(dim=1) / denom_k  # [K]

#         denom_all = denom_k.sum()
#         C_weighted_multi = (C_weighted_k * denom_k).sum() / denom_all

#         c_index_results = {
#             "C_k_t": C,                     # [K, T] on CPU
#             "C_integrated_k": C_integrated_k,       # [K]
#             "C_weighted_k": C_weighted_k,   # [K]
#             "C_weighted_multi": C_weighted_multi.item()  # scalar
#         }
#         return c_index_results
      
    # # Convert logits to probabilities using softmax over time dimension
    # # Shape: [batch_size, num_risks, num_time_bins]
    # # compute joint pmf
    # B, K, T = logits.shape
    # device = logits.device    
    
    # cif = calculate_cif(logits)    # [B, K, T]       
    
    # # ---------------------- C-index Calculation ----------------------
    # # For C-index, we need a risk score (higher = higher risk)
    # # Use predicted probability of event by the actual event time
    # # Sum over all risks and time bins up to observed time

    # C = torch.zeros(K, T)

    # for k in range(K):
    #     is_k = (event_label == k)             # [B]
        
    #     # Event times for cause k patients
    #     tt_k = time_to_event[is_k]              # [num_k]

    #     if tt_k.numel() == 0:
    #         continue  # no patients with this cause

    #     # CIF for cause k, shape [B, T]
    #     cif_k = cif[:, k, :]
        
    #     for t in range(T):
    #         # Patients who had event k at or before t
    #         # These are the "cases" in C-index
    #         cases = is_k & (time_to_event <= t)
    #         if cases.sum() == 0:
    #             continue

    #         # For each case i, we compare them to all others j
    #         # whose event time is >= event time of i OR had competing event
    #         #
    #         # Create pairwise comparison matrix:
    #         #
    #         # For vectorization, expand risk scores
    #         R_i = cif_k[cases, t].view(-1, 1)    # shape [Nc, 1]
    #         R_j = cif_k[:, t].view(1, -1)        # shape [1, B]
            

    #         # Valid comparators j:
    #         # j must not have event k before time of i
    #         tt_cases = time_to_event[cases].view(-1, 1)     # [Nc,1]
    #         valid_j = (time_to_event >= tt_cases) | (event_label != k)  # [Nc,B]
            
    #         # Concordance condition: R_i > R_j
    #         concordant = (R_i > R_j) & valid_j

    #         # Comparable pairs count
    #         comparable = valid_j.sum().float()

    #         if comparable > 0:
    #             C[k, t] = concordant.sum().float() / comparable
    #         else:
    #             C[k, t] = torch.nan
    
    # # integrated C-index
    # C_integrated_k = C.mean(dim=1)   

    # # integrated multiclass C-index

    # # event weighted
    # true_time_onehot = torch.zeros(B, T, device=device)
    # true_time_onehot.scatter_(1, (time_to_event).long().view(B,1), 1.0)  # [B,T]
    
    # true_cause_onehot = torch.zeros(B, K, device=device)
    # true_cause_onehot.scatter_(1, event_label.view(B,1).long(), 1.0)  # [B, K]
    
    # # ew cause specific
    # event_occ = true_cause_onehot.unsqueeze(2) * true_time_onehot.unsqueeze(1)  # [B, K, T]
    # E_k_t = event_occ.sum(dim=0)  # [K, T] number of events of cause k at time t
    # denom_k = E_k_t.sum(dim=1)  # [K]
    # C_weighted_k = (C * E_k_t).sum(dim=1) / denom_k   # [K]

    # # ew multiclass
    # denom_all = denom_k.sum()
    # C_weighted_multi = (C_weighted_k * denom_k).sum() / denom_all

    # c_index_results = {
    #     "C_k_t": C,                     # [K, T]
    #     "C_integrated_k": C_integrated_k,       # [K]
    #     "C_weighted_k": C_weighted_k,   # [K]
    #     "C_weighted_multi": C_weighted_multi  # scalar
    # }

    # return c_index_results


def calculate_brier_score(
    logits: torch.Tensor,
    time_to_event: torch.Tensor,
    event_label: torch.Tensor,
) -> Dict[str, float]:
     
    B, K, T = logits.shape
    device = logits.device      

    cif = calculate_cif(logits)    # [B, K, T]
    
    # ----------------------- Brier Score Calculation ----------------------
    # 3) Build true indicator tensors
    times = torch.arange(0, T, device=device).view(1, T)   # [1, T]

    # I_{i,k}(t) = 1 if true_time <= t and true_cause == k
    # Build shape [B, K, T] with broadcasting
    # mask_time = (time_to_event <= times) -> [B, T]
    mask_time = (time_to_event.unsqueeze(1) <= times)                                # [B, T]
    # mask_cause = (tc == k) -> we will expand it
    # create [B, K, 1] then broadcast
    cause_idx = torch.arange(K, device=device).view(1, K, 1)  # [1, K, 1]
    true_cause_expand = event_label.view(B, 1, 1)                     # [B, 1, 1]
    mask_cause = (true_cause_expand == cause_idx)           # [B, K, 1]

    I_ik_t = (mask_cause.expand(B, K, T) & mask_time.view(B, 1, T).expand(B, K, T)).float()  # [B, K, T]

    # 4) Cause-specific Brier per time BS_k(t)
    # squared errors: [B, K, T]
    sq_err = (I_ik_t - cif) ** 2
    bs_k_t = sq_err.mean(dim=0)   # average over B -> shape [K, T]

    # 5) Multiclass Brier per time is sum over causes
    bs_multi_t = bs_k_t.sum(dim=0)   # [T]

    # 6) Integrated (simple mean over times)
    ibs_k = bs_k_t.mean(dim=1)       # [K] mean over T
    ibs_multi = bs_multi_t.mean()    # scalar

    # event-weighted IBS
    # E(t) = number of events at exactly time t (any cause)
    # true_time==t and note every trajectory ends with an event
    # Build E(t): [T]
    true_time_onehot = torch.zeros(B, T, device=device)
    # index positions true_time-1 (if true_time in 1..T)
    true_time_onehot.scatter_(1, (time_to_event).long().view(B,1), 1.0)  # [B,T]
    
    true_cause_onehot = torch.zeros(B, K, device=device)
    true_cause_onehot.scatter_(1, event_label.view(B,1).long(), 1.0)  # [B, K]
    
    # 7) event specific ew IBS
    event_occ = true_cause_onehot.unsqueeze(2) * true_time_onehot.unsqueeze(1)  # [B, K, T]
    E_k_t = event_occ.sum(dim=0)  # [K, T] number of events of cause k at time t

    denom_k = E_k_t.sum(dim=1)  # [K]
    ew_ibs_k = (bs_k_t * E_k_t).sum(dim=1) / denom_k   # [K]
    
    # 8) multi-event ew IBS  
    E_t = true_time_onehot.sum(dim=0)   # [T] number of events at time t
    denom = E_t.sum()

    # weighted sums across times
    ew_ibs_multi = (bs_multi_t * E_t).sum() / denom   

    brier_results = {        
        "bs_k_t": bs_k_t,           # [K, T]
        "bs_multi_t": bs_multi_t,   # [T]
        "ibs_k": ibs_k,             # [K]
        "ibs_multi": ibs_multi,     # scalar
        "ew_ibs_k": ew_ibs_k,       # [K] or None
        "ew_ibs_multi": ew_ibs_multi  # scalar or None
    }

    return brier_results


def calculate_roc_and_prc(logits, time_to_event, event_label, return_full_curves: bool = False) -> Dict[str, float]:
    # simply calculate the classification as the sum over time of the predicted probabilities
    # for mortality (event_label == 1)

    B, K, T = logits.shape
    device = logits.device    
    
    cif = calculate_cif(logits)    # [B, K, T]
    # risk score for mortality (event 1)   
    risk_score_mortality = cif[:, 1, -1]   # [B]    

    y = event_label.cpu().numpy()
    p = risk_score_mortality.cpu().numpy()

    # implement ROC and PRC calculations as needed
    # AUROC (area under ROC curve)
    auroc = roc_auc_score(y, p)

    # Average Precision (area under PR curve)
    avg_precision = average_precision_score(y, p)
    prevalence = y.mean()  # equals proportion of positive samples

    output = {
        "auroc": auroc,
        "average_precision": avg_precision,
        "prevalence": prevalence
    }

    # ---- Optional full curves ----
    if return_full_curves:
        fpr, tpr, roc_thresholds = roc_curve(y, p)
        precision, recall, pr_thresholds = precision_recall_curve(y, p)

        output.update({
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": roc_thresholds,
            },
            "pr_curve": {
                "precision": precision,
                "recall": recall,
                "thresholds": pr_thresholds,
            }
        })

    return output


def compute_survival_metrics(
    logits: torch.Tensor,
    time_to_event: torch.Tensor,
    event_label: torch.Tensor,
    return_full_curves: bool = False,
    no_censoring: bool = True
) -> Dict[str, float]:
    """
    Compute survival analysis metrics. # Note that we have no censoring, and that 
    is assumed in the code below.
    
    Args:
        predictions: [batch_size, num_risks, num_time_bins] - predicted probabilities
        time_to_event: [batch_size] - actual time to event
        event_label: [batch_size] - event type (0=censored, 1=event1, 2=event2, etc.)
        time_bins: array of time bin edges
        
    Returns:
        Dictionary of survival metrics
calculate_roc_and_prc(logits, time_to_event, event_label)
    NOTE: Censoring is not handled in this implementation.
        
    NOTE: need to implement bootstrapping with confidence intervals for final analysis. 
    Will report integrated Brier score and event weighted integrated Brier score, both multiclass.
    Could potentially show curves of Brier over time as well.
    """

    with torch.inference_mode():
        # subtract 1 from event_label if no censoring
        event_label = event_label -1 if no_censoring else event_label   

        c_index_results = calculate_c_index(logits, time_to_event, event_label)

        brier_results = calculate_brier_score(logits, time_to_event, event_label)

        # calculate auroc and average precision for the mortality prediction
        roc_and_prc_results = calculate_roc_and_prc(logits, time_to_event, event_label, return_full_curves=return_full_curves)

        if return_full_curves:
            return {
                'weighted_C_index': c_index_results['C_weighted_k'],
                'weighted_multi_C': c_index_results['C_weighted_multi'],
                'ibs_weighted': brier_results['ew_ibs_k'], 
                'auroc': roc_and_prc_results['auroc'],
                'average_precision': roc_and_prc_results['average_precision'],
                'prevalence': roc_and_prc_results['prevalence'],
                'roc_curve': roc_and_prc_results['roc_curve'],
                'pr_curve': roc_and_prc_results['pr_curve']}
        else:
            return {
                'weighted_C_index': c_index_results['C_weighted_k'],
                'weighted_multi_C': c_index_results['C_weighted_multi'],
                'ibs_weighted': brier_results['ew_ibs_k'], 
                'auroc': roc_and_prc_results['auroc'],
                'average_precision': roc_and_prc_results['average_precision'],
                'prevalence': roc_and_prc_results['prevalence']}
        

def calculate_all_bin_ece(logits, true_bins, true_events, n_calibration_bins=10):
    """
    Calculates the Expected Calibration Error (ECE) for each event and time bin.
    
    Args:
        logits (np.ndarray or torch.Tensor): Shape [N, K, 8]
        true_bins (np.ndarray): Shape [N], values 0-7 (event time bin)
        true_events (np.ndarray): Shape [N], values 1 to K (event type label)
        n_calibration_bins (int): Number of bins for the reliability grouping (default 10)
        
    Returns:
        ece_matrix (np.ndarray): Shape [K, 8], where each cell is the ECE for that 
                                 event at that specific cumulative time horizon.
    """
    # 1. Get CIF [N, K, 8] using your joint-softmax logic
    cif_tensor = calculate_cif(logits)
    cif = cif_tensor.detach().cpu().numpy()
    
    N, K, T = cif.shape
    ece_matrix = np.zeros((K, T))
    # New: dictionary to store (mean_pred, mean_true, bin_counts) for plotting
    curve_data = {}
    bin_boundaries = np.linspace(0, 1, n_calibration_bins + 1)

    for k in range(K):
        for t in range(T):
            # Ground Truth: Did event 'k' happen at or before bin 't'?
            actual_outcomes = ((true_bins <= t) & (true_events == k)).astype(float)
            
            # Predicted cumulative probability for event k at time t
            pred_probs_t = cif[:, k, t]

            # Pre-initialize for 10 bins
            confidences = np.full(n_calibration_bins, np.nan)
            accuracies = np.full(n_calibration_bins, np.nan)
            counts = np.zeros(n_calibration_bins)
            
            ece_t = 0.0
            for i in range(n_calibration_bins):
                # Mask samples that fall into the current probability bin (e.g., 0.1 to 0.2)
                bin_mask = (pred_probs_t > bin_boundaries[i]) & (pred_probs_t <= bin_boundaries[i+1])
                
                bin_count = np.sum(bin_mask)
                if bin_count > 0:
                    # Average confidence: mean of predicted probabilities in bin
                    avg_conf = np.mean(pred_probs_t[bin_mask])
                    # Average accuracy: actual frequency of events in bin
                    avg_acc = np.mean(actual_outcomes[bin_mask])
                    
                    # Weighted absolute calibration gap
                    ece_t += (bin_count / N) * np.abs(avg_acc - avg_conf)

                    # Save for plotting
                    confidences[i] = avg_conf
                    accuracies[i] = avg_acc
                    counts[i] = bin_count
            
            ece_matrix[k, t] = ece_t
            
            curve_data[f"{k}_{t}"] = {
                "confidences": confidences,
                "accuracies": accuracies,
                "counts": counts
            }
            
    # calculate integrated and event weighted ECE
    K, T = ece_matrix.shape
    
    # 1. Integrate over Time (Average across the 8 bins)
    # This gives you one ECE value per event type
    event_eces = np.mean(ece_matrix, axis=1) # Shape: [K]

    # 2. Calculate Event Weights (Prevalence)
    # Count how many of each event occurred in the test set
    counts = np.array([np.sum(true_events == k) for k in range(K)])
    weights = counts / np.sum(counts)

    # 3. Final Weighted Scalar
    summary_ece = np.sum(event_eces * weights)

    return {
        "ece_matrix": ece_matrix,
        "event_eces": event_eces,
        "summary_ece": summary_ece,
        "curve_data": curve_data
    }
    
