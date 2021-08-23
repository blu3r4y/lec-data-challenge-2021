from typing import List

import numpy as np
import pandas as pd


def splits(x: pd.DataFrame, y: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """Get all possible data splits"""
    x_train, y_train = x.loc[masks["train"]], y.loc[masks["train"]]
    x_val, y_val = x.loc[masks["val"]], y.loc[masks["val"]]

    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

    return x_train, x_val, y_train, y_val


def combine_features(*args: pd.DataFrame) -> pd.DataFrame:
    """Horizontally stack multiple features together"""
    nrows = np.array([df.shape[0] for df in args])
    assert np.all(nrows == nrows[0])

    return pd.concat(
        args, axis="columns", verify_integrity=True
    )  # check for duplicates


def drop_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop the selected features from the data frame"""
    return df.drop(columns, axis="columns", errors="raise")


def select_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Only select the pressure column into a new dataframe"""
    assert "pa" in df
    return df["pa"].to_frame()


def select_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Only select the cycle column into a new dataframe"""
    assert "cycle" in df
    return df["cycle"].to_frame()


def merge_pressure(df: pd.DataFrame, pressure: pd.DataFrame) -> pd.DataFrame:
    """Horizontally stack the raw test file and the mapped predictions together"""
    assert df.shape[0] == pressure.shape[0]

    # hack to quote the column manually in the output
    df2 = df.copy(deep=False)
    df2["cycle"] = df2["cycle"].map(lambda e: f'"{e}"')

    return pd.concat([df2, pressure], axis="columns", verify_integrity=True)
