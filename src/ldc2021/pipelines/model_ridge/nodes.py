import logging
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ...utils.evaluation import report_scores, splits

log = logging.getLogger(__name__)


def train(
    x: pd.DataFrame, y: pd.Series, masks: pd.DataFrame, ridge_args: dict,
) -> Ridge:
    x_train, x_val, y_train, y_val = splits(x, y, masks)

    log.info(f"ridge regression hyperparameters: {ridge_args}")

    # train model
    clf = Ridge(**ridge_args)
    clf.fit(x_train, y_train)

    report_scores(partial(predict, model=clf), x_train, x_val, y_train, y_val)

    return clf


def predict(model: Ridge, x: np.ndarray) -> np.ndarray:
    y = model.predict(x).ravel()
    return pd.Series(y, name="pa").to_frame()
