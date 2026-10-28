from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)


def eer_score(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    idx = np.where(diff >= 0)[0]
    if len(idx) == 0:
        return 0.0
    i = int(idx[0])
    if i == 0:
        return float((fpr[0] + fnr[0]) / 2.0)
    d_prev, d_curr = diff[i - 1], diff[i]
    if d_curr == d_prev:
        eer = (fpr[i] + fnr[i]) / 2.0
    else:
        weight = -d_prev / (d_curr - d_prev)
        eer_fpr = fpr[i - 1] + weight * (fpr[i] - fpr[i - 1])
        eer_fnr = fnr[i - 1] + weight * (fnr[i] - fnr[i - 1])
        eer = (eer_fpr + eer_fnr) / 2.0
    return float(np.clip(eer, 0.0, 1.0))

def tar_at_far(y_true, scores, far=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(fpr <= far)[0]
    return float(tpr[valid[-1]]) if len(valid) else 0.0


def expected_calibration_error(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true)
    probs = np.asarray(probabilities)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if mask.any():
            conf = probs[mask].mean()
            acc = y_true[mask].mean()
            ece += mask.mean() * abs(conf - acc)
    return float(ece)


def binary_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)
    probs = np.clip(scores, 0, 1)
    return {
        "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, preds)),
        "eer": eer_score(y_true, scores),
        "tar_at_far_1pct": tar_at_far(y_true, scores, 0.01),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
    }
