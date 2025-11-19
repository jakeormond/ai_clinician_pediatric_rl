import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, \
    average_precision_score, roc_curve, precision_recall_curve

from typing import Dict


import logging
logger = logging.getLogger(__name__)


def calculate_cif(
    logits: torch.Tensor
    ):
    
    B, K, T = logits.shape
    logits_flat = logits.view(B, K * T)   
    p = F.softmax(logits_flat, dim=1).view(B, K, T)   # [B, K, T]

    # 2) CIF: cumulative over time
    cif = p.cumsum(dim=2)  

    return cif


def calculate_c_index(
    logits: torch.Tensor,
    time_to_event: torch.Tensor,
    event_label: torch.Tensor,
) -> Dict[str, float]:
      
    # Convert logits to probabilities using softmax over time dimension
    # Shape: [batch_size, num_risks, num_time_bins]
    # compute joint pmf
    B, K, T = logits.shape
    device = logits.device    
    
    cif = calculate_cif(logits)    # [B, K, T]       
    
    # ---------------------- C-index Calculation ----------------------
    # For C-index, we need a risk score (higher = higher risk)
    # Use predicted probability of event by the actual event time
    # Sum over all risks and time bins up to observed time

    C = torch.zeros(K, T)

    for k in range(K):
        is_k = (event_label == k)             # [B]
        
        # Event times for cause k patients
        tt_k = time_to_event[is_k]              # [num_k]

        if tt_k.numel() == 0:
            continue  # no patients with this cause

        # CIF for cause k, shape [B, T]
        cif_k = cif[:, k, :]
        
        for t in range(T):
            # Patients who had event k at or before t
            # These are the "cases" in C-index
            cases = is_k & (time_to_event <= t)
            if cases.sum() == 0:
                continue

            # For each case i, we compare them to all others j
            # whose event time is >= event time of i OR had competing event
            #
            # Create pairwise comparison matrix:
            #
            # For vectorization, expand risk scores
            R_i = cif_k[cases, t].view(-1, 1)    # shape [Nc, 1]
            R_j = cif_k[:, t].view(1, -1)        # shape [1, B]
            

            # Valid comparators j:
            # j must not have event k before time of i
            tt_cases = time_to_event[cases].view(-1, 1)     # [Nc,1]
            valid_j = (time_to_event >= tt_cases) | (event_label != k)  # [Nc,B]
            
            # Concordance condition: R_i > R_j
            concordant = (R_i > R_j) & valid_j

            # Comparable pairs count
            comparable = valid_j.sum().float()

            if comparable > 0:
                C[k, t] = concordant.sum().float() / comparable
            else:
                C[k, t] = torch.nan
    
    # integrated C-index
    C_integrated_k = C.mean(dim=1)   

    # integrated multiclass C-index

    # event weighted
    true_time_onehot = torch.zeros(B, T, device=device)
    true_time_onehot.scatter_(1, (time_to_event).long().view(B,1), 1.0)  # [B,T]
    
    true_cause_onehot = torch.zeros(B, K, device=device)
    true_cause_onehot.scatter_(1, event_label.view(B,1).long(), 1.0)  # [B, K]
    
    # ew cause specific
    event_occ = true_cause_onehot.unsqueeze(2) * true_time_onehot.unsqueeze(1)  # [B, K, T]
    E_k_t = event_occ.sum(dim=0)  # [K, T] number of events of cause k at time t
    denom_k = E_k_t.sum(dim=1)  # [K]
    C_weighted_k = (C * E_k_t).sum(dim=1) / denom_k   # [K]

    # ew multiclass
    denom_all = denom_k.sum()
    C_weighted_multi = (C_weighted_k * denom_k).sum() / denom_all

    c_index_results = {
        "C_k_t": C,                     # [K, T]
        "C_integrated_k": C_integrated_k,       # [K]
        "C_weighted_k": C_weighted_k,   # [K]
        "C_weighted_multi": C_weighted_multi  # scalar
    }

    return c_index_results


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

    # subtract 1 from event_label if no censoring
    event_label = event_label -1 if no_censoring else event_label   

    c_index_results = calculate_c_index(logits, time_to_event, event_label)

    brier_results = calculate_brier_score(logits, time_to_event, event_label)

    # calculate auroc and average precision for the mortality prediction
    roc_and_prc_results = calculate_roc_and_prc(logits, time_to_event, event_label, return_full_curves=return_full_curves)

    if return_full_curves:
        return {'weighted_C_index': c_index_results['C_weighted_k'],
            'ibs_weighted': brier_results['ew_ibs_k'], 
            'auroc': roc_and_prc_results['auroc'],
            'average_precision': roc_and_prc_results['average_precision'],
            'prevalence': roc_and_prc_results['prevalence'],
            'roc_curve': roc_and_prc_results['roc_curve'],
            'pr_curve': roc_and_prc_results['pr_curve']}
    else:
        return {'weighted_C_index': c_index_results['C_weighted_k'],
            'ibs_weighted': brier_results['ew_ibs_k'], 
            'auroc': roc_and_prc_results['auroc'],
            'average_precision': roc_and_prc_results['average_precision'],
            'prevalence': roc_and_prc_results['prevalence']}