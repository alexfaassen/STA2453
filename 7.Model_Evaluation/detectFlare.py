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


def binary_to_intervals(y_pred):
    """
    Convert a binary (0/1) array 'y_pred' into a list of intervals (start_t, end_t),
    by grouping consecutive 1's. 'tt' is the time array aligned with y_pred.
    
    Example:
      tt = [0.0, 0.1, 0.2, 0.3, 0.4]
      y_pred = [0, 1, 1, 0, 1]
      => intervals = [(0.1, 0.2), (0.4, 0.4)]
    """

    tt = np.arange(y_pred.shape[0])

    intervals = []
    i = 0
    n = len(tt)
    while i < n:
        if y_pred[i] == 1:
            start_idx = i
            # move forward while we still have 1's
            while i < n and y_pred[i] == 1:
                i += 1
            end_idx = i - 1
            intervals.append((tt[start_idx], tt[end_idx]))
        else:
            i += 1
    return intervals


def event_level_scores(real_flares, y_pred):
    """
    Compute event-level Precision, Recall, F1 for flare detection when
    'y_pred' is a binary array of length == len(tt) (e.g., anomalies from an Isolation Forest).
    
    real_flares: list of (start_time, end_time) ground-truth intervals
    tt: array of times corresponding 1-to-1 with y_pred
    y_pred: 0/1 predictions for each time in 'tt'
    
    Steps:
      1) Convert y_pred => list of detected intervals (predicted_flares).
      2) For each real flare, see if it overlaps ANY predicted interval => True Positive.
      3) For each predicted interval, see if it overlaps ANY real flare => Not a False Positive.
      4) Compute precision, recall, and F1 using TP, FP, FN at the event level.
    """
    # Convert your binary detection array into intervals
    predicted_flares = binary_to_intervals(y_pred)
    
    # True Positives: real flares matched by at least one predicted flare
    matched_real = 0
    for rf in real_flares:
        if any(flares_overlap(rf, pf) for pf in predicted_flares):
            matched_real += 1
    
    TP = matched_real
    FN = len(real_flares) - TP  # flares that had no overlap
    # Among predicted flares, see which overlap any real flare
    matched_pred = 0
    for pf in predicted_flares:
        if any(flares_overlap(pf, rf) for rf in real_flares):
            matched_pred += 1
    FP = len(predicted_flares) - matched_pred

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    return precision, recall, f1
