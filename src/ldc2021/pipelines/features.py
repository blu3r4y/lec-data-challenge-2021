from typing import List, Optional

import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node
from scipy.integrate import cumtrapz


def create_pipeline():
    return Pipeline(
        [
            node(log_acceleration, ["clean"], "log"),
            node(sqrt_acceleration, ["clean"], "sqrt"),
            node(
                compute_velocities, ["clean", "params:velocity.absolute"], "velocity",
            ),
            node(
                compute_rolling_std,
                ["clean", "params:rolling.sizes", "params:rolling.type"],
                "rolling",
            ),
        ]
    )


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


def compute_velocities(df: pd.DataFrame, absolute: bool = False) -> pd.DataFrame:
    """Integrate the acceleration over the angle.

    :param df: The dataframe
    :param absolute: Also integrate the absolute velocities, defaults to False
    """
    df1 = _compute_velocity(df, absolute=False)

    if absolute:
        df2 = _compute_velocity(df, absolute=True)

    return pd.concat([df1, df2], axis=1)


def _compute_velocity(df: pd.DataFrame, absolute: bool = False) -> pd.DataFrame:
    name = f"vel{'_abs' if absolute else ''}"

    def _compute_velocity_per_cycle(grp: pd.DataFrame):
        result = pd.DataFrame(index=grp.index)

        acc, phi = grp["acc"], grp["phi"]
        if absolute:
            acc = acc.abs()

        vel = cumtrapz(acc, phi)
        vel = np.insert(vel, 0, 0)

        result[name] = vel
        return result

    # compute velocity by integrating acceleration per cycle
    return df.groupby("cycle").apply(_compute_velocity_per_cycle)


def compute_rolling_std(
    df: pd.DataFrame, window_sizes: List[int], window_type: Optional[str] = None
) -> pd.DataFrame:
    """Compute a rolling standard deviation.

    :param df: The dataframe
    :param window_sizes: The sizes of the rolling windows
    :param window_type: The window type, defaults to None
    """
    args = dict(min_periods=1, center=True, win_type=window_type)
    appx = f"_{window_type}" if window_type else ""

    def _compute_rolling_std(grp: pd.DataFrame):
        result = pd.DataFrame(index=grp.index)

        for w in window_sizes:
            result[f"roll_{w}_std{appx}"] = grp["acc"].rolling(window=w, **args).std()

        return result

    # compute rolling std per cycle
    return df.groupby("cycle").apply(_compute_rolling_std)
