from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest


def STLIF(data, contamination, detrend=True, n_estimators=100, sample_size=256):
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
        stl = STL(data, period=240, robust=True)  # Use period=240 based on EDA
        decomposition = stl.fit()
        data = decomposition.resid.to_frame()

        # decomposition.plot()


    ## Run Model
    # Isolation Forest
    model = IsolationForest(n_estimators = n_estimators, contamination = contamination, max_samples = sample_size)
    #contamination = 'auto', random_state = 42)
    # random_state: for reproducibility.
    train = data[[data.columns[0]]].values
    model.fit(train) #data

    # Predict anomalies
    anomalies = model.predict(train)
    scores = model.decision_function(train)

    # Save
    data['anomaly'] = anomalies
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})
    data['anomaly_score'] = scores

    return data