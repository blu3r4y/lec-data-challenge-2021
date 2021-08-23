from typing import List, Optional

import pandas as pd


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
