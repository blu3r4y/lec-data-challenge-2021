import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ...utils.columns import splits

log = logging.getLogger(__name__)


def train(
    x: pd.DataFrame, y: pd.Series, masks: pd.DataFrame, ridge_args: dict,
) -> Ridge:
    x_train, x_val, y_train, y_val = splits(x, y, masks)

    log.info(f"ridge regression hyperparameters: {ridge_args}")

    # train model
    clf = Ridge(**ridge_args)
    clf.fit(x_train, y_train)

    # check train scores
    y_train_pred = predict(clf, x_train)
    score_train = mean_squared_error(y_train, y_train_pred, squared=False)

    # check validation scores
    y_val_pred = predict(clf, x_val)
    score_val = mean_squared_error(y_val, y_val_pred, squared=False)

    log.info(f"RMSE [train] : {score_train}")
    log.info(f"RMSE [val]   : {score_val}")
    log.info("=" * 80)

    return clf


def predict(model: Ridge, x: np.ndarray) -> np.ndarray:
    y = model.predict(x).ravel()
    return pd.Series(y, name="pa")
