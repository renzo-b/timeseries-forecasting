import pandas as pd


def cross_validate(ts, label_columns, model, splitter):
    """Cross validates timeseries data 
    
    Inputs
    ------
    ts: array
        timeseries data
    
    label_columns: list

    model

    splitter

    Returns
    -------
    cv_results: df
        cross validation results
    """
    cv_results = []

    # split into train and validation
    for train_index, validate_index in splitter.split(ts):
        # fit using only training data
        model.fit(ts.iloc[train_index], label_columns)
        y_hat = model.predict()  # prediction
        y_true = ts[validate_index]  # actual

        # make a dataframe of results for this loop
        df_loop = pd.concat([y_hat, y_true], axis=1)
        df_loop.columns = ["y_hat", "y_true"]
        df_loop["cutoff"] = ts.index[train_index][-1]
        df_loop["future dates"] = ts.index[validate_index]
        cv_results.append(df_loop.reset_index(drop=True))

    # concatenate all the results into a single dataframe
    cv_results = pd.concat(cv_results, axis=0)

    print(cv_results)
    return cv_results

