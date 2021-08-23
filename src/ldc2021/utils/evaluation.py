import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

log = logging.getLogger(__name__)


def splits(x: pd.DataFrame, y: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """Get all possible data splits"""
    x_train, y_train = x.loc[masks["train"]], y.loc[masks["train"]]
    x_val, y_val = x.loc[masks["val"]], y.loc[masks["val"]]

    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

    return (
        x_train.to_numpy(),
        x_val.to_numpy(),
        y_train.to_numpy().ravel(),
        y_val.to_numpy().ravel(),
    )


def report_scores(
    predict_fun: Callable,
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
):
    """Report the scores on the training and validation sets"""
    # check train scores
    y_train_pred = predict_fun(x=x_train)
    score_train = mean_squared_error(y_train, y_train_pred, squared=False)

    # check validation scores
    y_val_pred = predict_fun(x=x_val)
    score_val = mean_squared_error(y_val, y_val_pred, squared=False)

    log.info(f"RMSE [train] : {score_train}")
    log.info(f"RMSE [val]   : {score_val}")
    log.info("=" * 80)
