import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def cross_validate(ts, model, n_splits: int, test_size: int, gap: int = 0):
    """Cross validates timeseries data 
    
    Inputs
    ------
    ts: array
        timeseries data
    
    n_splits: int

    test_size: int

    gap:  int

    Returns
    -------
    cv_results: df
        cross validation results
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    cv_results = []

    # split into train and test
    for train_index, test_index in splitter.split(ts):
        # fit using only training data
        model.fit(ts.iloc[train_index])
        y_hat = model.predict()  # prediction
        y_true = ts[test_index]  # actual

        # make a dataframe of results for this loop
        df_loop = pd.concat([y_hat, y_true], axis=1)
        df_loop.columns = ["y_hat", "y_true"]
        df_loop["cutoff"] = ts.index[train_index][-1]
        df_loop["future dates"] = ts.index[test_index]
        cv_results.append(df_loop.reset_index(drop=True))

    # concatenate all the results into a single dataframe
    cv_results = pd.concat(cv_results, axis=0)

    print(cv_results)
    return cv_results

