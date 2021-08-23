import logging
from functools import partial

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from ...utils.evaluation import report_scores, splits

log = logging.getLogger(__name__)

CYCLE_LENGTH = 601


def train(
    x: pd.DataFrame, y: pd.Series, masks: pd.DataFrame, mlp_args: dict
) -> MLPRegressor:
    x_train, x_val, y_train, y_val = splits(x, y, masks)

    log.info(f"mlp hyperparameters: {mlp_args}")

    clf = MLPRegressor(**mlp_args)
    clf.fit(*reshape(x_train, y_train))

    report_scores(partial(predict, model=clf), x_train, x_val, y_train, y_val)

    return clf


def predict(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    y = model.predict(reshape(x))
    return pd.DataFrame(y.reshape(-1, 1))


def reshape(x_train, y_train=None):
    shape_both = y_train is not None
    n_features = x_train.shape[1]

    # ensure that we have numpy arrays that we can shape
    if not isinstance(x_train, np.ndarray):
        x_train = x_train.to_numpy()
    if shape_both and not isinstance(y_train, np.ndarray):
        y_train = y_train.to_numpy()

    # shape an entire cycle into the model
    x_train_shp = x_train.reshape(-1, CYCLE_LENGTH * n_features)
    y_train_shp = y_train.reshape(-1, CYCLE_LENGTH) if shape_both else None

    log.info(f"Reshaped x values to {x_train_shp.shape}")

    return (x_train_shp, y_train_shp) if shape_both else x_train_shp
