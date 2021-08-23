from kedro.pipeline import Pipeline, node

from ...utils.columns import merge_pressure
from ...utils.selection import get_feature_selection_for
from .nodes import predict, train


def create_pipeline():
    # DO NOT namespace this pipeline
    return Pipeline(
        [
            # select features for both train and test sets
            node(
                get_feature_selection_for("train"),
                ["params:selection.minimal"],
                "train.input_cache",
            ),
            node(
                get_feature_selection_for("test"),
                ["params:selection.minimal"],
                "test.input_cache",
            ),
            # training
            node(
                train,
                ["train.input_cache", "train.pressure", "masks", "params:ridge"],
                "model_ridge",
            ),
            # inference
            node(
                predict, ["model_ridge", "train.input_cache"], "train.model_ridge_out"
            ),
            node(predict, ["model_ridge", "test.input_cache"], "test.model_ridge_out"),
            # final dataset
            node(
                merge_pressure,
                ["test.raw", "test.model_ridge_out"],
                "submission_ridge",
            ),
        ]
    )
