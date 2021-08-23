import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

log = logging.getLogger(__name__)


def train(x_train: pd.DataFrame, y_train: pd.Series, ridge_args: dict) -> Ridge:
    log.info(f"ridge regression hyperparameters: {ridge_args}")
    clf = Ridge(**ridge_args)
    clf.fit(x_train, y_train)

    # check train score
    y_pred = predict(clf, x_train.to_numpy())
    score = mean_squared_error(y_train.to_numpy(), y_pred.to_numpy(), squared=False)
    log.info(f"RMSE: {score}")
    log.info("=" * 80)

    return clf


def predict(model: Ridge, x: np.ndarray) -> np.ndarray:
    y = model.predict(x).ravel()
    return pd.Series(y, name="pa")
