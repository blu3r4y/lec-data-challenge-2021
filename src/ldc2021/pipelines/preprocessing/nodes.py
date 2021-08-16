import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz


def transform_cycle(df: pd.DataFrame) -> pd.DataFrame:
    # remove prefix (either "train" or "test")
    prefix_length = len(df["cycle"].iloc[0].split("0")[0])
    df["cycle"] = df["cycle"].map(lambda e: int(e[prefix_length:]))

    return df


def compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    def _compute_velocity_per_cycle(grp: pd.DataFrame):
        vel = cumtrapz(grp["acc"], grp["phi"])
        vel = np.insert(vel, 0, 0)

        grp["vel"] = vel
        return grp

    # compute velocity by integrating acceleration per cycle
    df = df.groupby("cycle").apply(_compute_velocity_per_cycle)

    return df
