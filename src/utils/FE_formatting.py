import os
import sys
import hashlib
import pickle

# For data science in general
import numpy as np
import pandas as pd


def expand_column_to_columns(df_or, param='ifc'):
    """
    Expands a column in a DataFrame containing iterable elements into separate columns.

    Parameters
    ----------
    df_or : pandas.DataFrame
        The input DataFrame that contains the column to be expanded.
    param : str, optional, default='ifc'
        The name of the column in df_or to expand. Each element in this column
        should be iterable (like a list or numpy array) of the same length.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified column expanded into multiple columns.
        The new columns are named as '{param}_0', '{param}_1', ..., '{param}_{n-1}',
        where n is the length of the iterable in the first row of the specified column.
        The original column is preserved in the returned DataFrame.

    Notes
    -----
    - If any element in the column is None, it will be replaced with an array of NaNs.
    """
    df = df_or.copy()
    len_param = len(df.iloc[0][param])
    param_nm = param + '_copy'
    df[param_nm] = df[param]
    df[param_nm] = df[param_nm].apply(lambda x: x if x is not None else np.full(len_param, np.nan))
    df_expanded = pd.DataFrame(df[param_nm].tolist(), columns=[f'{param}_{i}' for i in range(len_param)])
    df_merged_large = pd.concat([df.drop(columns=[param_nm]).reset_index(drop=True), df_expanded], axis=1)
    return df_merged_large
