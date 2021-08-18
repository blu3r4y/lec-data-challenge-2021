import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node
from scipy.integrate import cumtrapz


def create_pipeline():
    return Pipeline([node(compute_velocity, ["clean"], "features")])


def compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    def _compute_velocity_per_cycle(grp: pd.DataFrame):
        vel = cumtrapz(grp["acc"], grp["phi"])
        vel = np.insert(vel, 0, 0)

        grp["vel"] = vel
        return grp

    # compute velocity by integrating acceleration per cycle
    df = df.groupby("cycle").apply(_compute_velocity_per_cycle)

    return df
