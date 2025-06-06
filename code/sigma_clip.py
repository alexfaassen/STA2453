"""sigma_clip.py

Implements a simple a-bsigma-clipping model to detect flares or outliers
in 1D time series data.
"""

import numpy as np
from typing import Union

def sigma_clip(
    y: Union[np.ndarray, list],
    sigma: float = 3.0,
    consecutive_pts: int = 3
) -> np.ndarray:
    """
    Detect flares (or outliers) in array y via a basic sigma-clipping rule.
    
    Parameters
    ----------
    y : array-like
        The 1D data (time series) to examine.
    sigma : float
        The threshold (in standard deviations) for clipping. 
        e.g., sigma=3 means |y - mean(y)| > 3 * std(y) are flagged.
    consecutive_pts : int
        Require this many consecutive outliers to label them as a flare.
        For example, consecutive_pts=1 means any single outlier is flagged
        as a flare, whereas consecutive_pts=2 means we need at least two
        consecutive outliers.
    
    Returns
    -------
    flares : ndarray of int (0 or 1)
        A binary array of the same length as y, where 1 indicates
        the data point is flagged as flare/outlier, and 0 indicates normal.
    """
    ## Initialize
    # data = y.copy()

    y = np.asarray(y)
    ## Compute overall mean & std
    mu = np.mean(y)
    sd = np.std(y)
    
    ## Create a mask of outliers: True if outside +/- sigma * sd
    mask_outlier = np.abs(y - mu) > sigma * sd
    
    ## Optional: Enforce 'consecutive_pts' requirement
    flares = np.zeros_like(y, dtype=int)
    if consecutive_pts <= 1:
        flares[mask_outlier] = 1
    else:
        # Scan for blocks of 'consecutive_pts' True in mask_outlier
        i = 0
        n = len(y)
        while i < n:
            if mask_outlier[i]:
                # Check if from i to i+consecutive_pts-1 are all True
                block_end = i + consecutive_pts
                if block_end <= n and all(mask_outlier[i:block_end]):
                    # Mark them as flares
                    flares[i:block_end] = 1
                    i = block_end
                else:
                    i += 1
            else:
                i += 1

    # # Save
    # data['anomaly'] = flares.ravel()
    # data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})
    # data['anomaly_score'] = scores

    return flares