import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz


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
