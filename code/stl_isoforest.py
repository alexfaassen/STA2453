"""stl_isoforest.py

Contains function to apply STL decomposition and fit an Isolation Forest
model for anomaly detection in brightness time series data.
"""

from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import pandas as pd
from typing import Union

# STL period based on EDA (e.g., known cadence = 2 min, daily = 720; periodicity ~4 hrs = 240)
STL_PERIOD = 240

def STLIF(
    data: pd.DataFrame,
    contamination: Union[float, str],
    detrend: bool = True,
    n_estimators: int = 100,
    sample_size: int = 256
) -> pd.DataFrame:
    """
    Applies STL decomposition (optional) and runs Isolation Forest for anomaly detection.

    Parameters:
        data (pd.DataFrame): Time series data with one column (brightness).
        contamination (float): The expected proportion of outliers.
        detrend (bool): Whether to apply STL decomposition to isolate residuals.
        n_estimators (int): Number of trees in the Isolation Forest.
        sample_size (int): Subsample size for each tree.

    Returns:
        pd.DataFrame: Input data with added columns for anomaly labels and scores.
    """

    ## Initialize
    data = data.copy()

    ## STL Decomposition
    if detrend:
        stl = STL(data, period=STL_PERIOD, robust=True)  # Use period=240 based on EDA
        decomposition = stl.fit()
        data = decomposition.resid.to_frame()

    ## Isolation Forest Model
    model = IsolationForest(
        n_estimators = n_estimators,
        contamination = contamination,
        max_samples = sample_size
    )
    train = data[[data.columns[0]]].values
    model.fit(train)

    # Predict anomalies
    anomalies = model.predict(train)
    scores = model.decision_function(train)

    # Save
    data['anomaly'] = anomalies
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})
    data['anomaly_score'] = scores

    return data