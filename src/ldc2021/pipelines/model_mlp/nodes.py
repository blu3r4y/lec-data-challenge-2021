import logging
from functools import partial

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from ...utils.evaluation import report_scores, splits

log = logging.getLogger(__name__)


def train(
    x: pd.DataFrame, y: pd.Series, masks: pd.DataFrame, mlp_args: dict
) -> MLPRegressor:
    x_train, x_val, y_train, y_val = splits(x, y, masks)

    log.info(f"mlp hyperparameters: {mlp_args}")

    clf = MLPRegressor(**mlp_args)
    clf.fit(x_train, y_train)

    report_scores(partial(predict, model=clf), x_train, x_val, y_train, y_val)

    return clf


def predict(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    y = model.predict(x).ravel()
    return pd.Series(y, name="pa").to_frame()
