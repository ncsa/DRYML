import pandas as pd
import numpy as np


def prep_df(DF, feature_list, index=True):
    """
    Returns a numpy of shape (ex, features) from a pandas dataframe
    """

    # Select features we're interested in.
    sel_DF = DF[feature_list].dropna()

    if index:
        return sel_DF.to_numpy(), sel_DF.index
    else:
        return sel_DF.to_numpy()


def prep_df_lags(DF, feature_list, lags, index=True):
    """
    Returns a numpy of shape (ex, lags, features) from a pandas dataframe
    """

    # Build lag list if lags were specified as an integer
    if type(lags) is int:
        lags = list(range(lags+1))[1:]

    # Stack lagged slices
    res_npy = []
    for feature in feature_list:
        lagged_features = []
        for lag in lags:
            lagged_features.append(DF[feature].shift(lag))
        lag_features = pd.concat(lagged_features, axis=1).dropna()
        res_npy.append(lag_features)

    # Build a common index
    common_idx = res_npy[0].index
    for feat in res_npy:
        common_idx = common_idx.intersection(feat.index)

    # Normalize the index
    for i in range(len(res_npy)):
        res_npy[i] = res_npy[i].loc[common_idx].to_numpy()

    # Build result array
    res_npy = np.stack(res_npy, axis=1)

    # Return dataset and possibly index
    if index:
        return res_npy, common_idx
    else:
        return res_npy
