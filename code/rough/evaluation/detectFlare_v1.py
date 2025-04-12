import numpy as np

def flares_overlap(true_flare, detected_flare):
    """
    Returns True if there's any time-overlap between a 'true_flare' and a 'detected_flare'.
    Each flare is a tuple (start_time, end_time).
    """
    t0_true, t1_true = true_flare
    t0_detected, t1_detected = detected_flare
    
    # They do NOT overlap if one ends before the other begins.
    no_overlap = (t1_true < t0_detected) or (t1_detected < t0_true)
    return not no_overlap


def event_level_scores(real_flares, detected_flares):
    """
    Computes precision, recall, and F1 score at the 'flare event' level.
    We say a real flare is 'found' (TP) if it overlaps with ANY detected flare.
    A detected flare is a FP if it doesn't overlap any real flare.

    :param real_flares:     list of (start_time, end_time) for actual flares
    :param detected_flares: list of (start_time, end_time) for flares from the model
    :return: (precision, recall, f1)
    """
    
    # Count how many real flares are matched by at least one detection
    matched_real = 0
    for rf in real_flares:
        # Check if there's ANY overlap with the detections
        overlaps_any = any(flares_overlap(rf, df) for df in detected_flares)
        if overlaps_any:
            matched_real += 1

    TP = matched_real
    FN = len(real_flares) - TP  # those not matched at all

    # Among the model’s detected flares, see which ones overlap any real flare
    matched_detected = 0
    for df in detected_flares:
        overlaps_any = any(flares_overlap(df, rf) for rf in real_flares)
        if overlaps_any:
            matched_detected += 1

    FP = len(detected_flares) - matched_detected

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
