import numpy as np
import pandas as pd


def log_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute `log(2 + acc)` of the acceleration.

    :param df: The dataframe
    """
    assert np.all(df["acc"] > -2)

    result = pd.DataFrame(index=df.index)

    result["acc_log2p"] = np.log(2 + df["acc"])
    result["acc_abs_log1p"] = np.log1p(np.abs(df["acc"]))

    return result


def sqrt_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Take the square root of the acceleration.

    :param df: The dataframe
    """
    result = pd.DataFrame(index=df.index)

    result["acc_abs_sqrt"] = np.sqrt(np.abs(df["acc"]))

    return result
