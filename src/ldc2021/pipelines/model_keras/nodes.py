import json
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import AlphaDropout, Dense

from ...utils.evaluation import report_scores, splits

log = logging.getLogger(__name__)


def train(x: pd.DataFrame, y: pd.Series, masks: pd.DataFrame):
    x_train, x_val, y_train, y_val = splits(x, y, masks)

    config = get_model_configuration(x_train)

    # compile model
    model = tf.keras.models.model_from_json(json.dumps(config["model"]))
    model.compile(
        **config["learning_args"],
        optimizer=tf.keras.optimizers.Adam(**config["adam_args"])
    )

    # configure callbacks
    stopper = tf.keras.callbacks.EarlyStopping(**config["early_stopping"], verbose=1)
    reducer = tf.keras.callbacks.ReduceLROnPlateau(
        **config["learning_rate_reducer"], verbose=1
    )

    # prepare results
    results = {"train": [], "test": [], "val": []}

    # train the model
    train = model.fit(
        x_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(x_val, y_val),
        verbose=2,
        callbacks=[stopper, reducer],
    )

    results["train"].append(train.history["acc"][-1])
    results["val"].append(train.history["val_acc"][-1])

    report_scores(lambda x: model.predict(x), x_train, x_val, y_train, y_val)

    return model


def predict(model, x: np.ndarray) -> np.ndarray:
    pass


def get_model_configuration(df: pd.DataFrame) -> dict:
    epochs = 50
    batch_size = 128

    early_stopping = {"monitor": "val_mse", "min_delta": 0, "patience": 100}
    learning_rate_reducer = {
        "monitor": "val_mse",
        "patience": 150,
        "factor": 0.5,
        "min_lr": 0.001,
    }

    input_dim = df.shape[1]

    model = tf.keras.models.Sequential()
    model.add(Dense(units=input_dim, activation="sigmoid", input_dim=input_dim))
    model.add(
        Dense(
            units=100,
            activation="tanh",
            kernel_initializer="lecun_normal",
            bias_initializer="lecun_normal",
        )
    )
    model.add(AlphaDropout(0.2))
    model.add(
        Dense(
            units=50,
            activation="tanh",
            kernel_initializer="lecun_normal",
            bias_initializer="lecun_normal",
        )
    )
    model.add(Dense(units=1, activation="sigmoid"))
    model = json.loads(model.to_json())

    learning_args = {
        "loss": "mean_squared_error",
        "metrics": ["mse"],
    }
    adam_args = {"learning_rate": 3e-4}

    return locals()
