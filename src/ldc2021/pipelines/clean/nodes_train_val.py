from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_mask(
    df: pd.DataFrame, split_args: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get a training and validation split based on the cycle identifier"""
    num_cycles = df["cycle"].max()
    cycles = np.arange(1, num_cycles + 1)

    idx_train, idx_val = train_test_split(cycles, **split_args)

    mask_train = df["cycle"].isin(set(idx_train))
    mask_val = df["cycle"].isin(set(idx_val))

    # all cycles are used
    assert np.all(mask_train ^ mask_val)

    masks = pd.DataFrame({"train": mask_train, "val": mask_val}, index=df.index)
    return masks
