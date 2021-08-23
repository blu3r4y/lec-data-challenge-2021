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
                ["params:selection.neural"],
                "train.input_cache",
            ),
            node(
                get_feature_selection_for("test"),
                ["params:selection.neural"],
                "test.input_cache",
            ),
            # training
            node(
                train,
                ["train.input_cache", "train.pressure", "masks"],
                "model_mlp_deep",
            ),
            # inference
            # node(predict, ["model_mlp_deep", "train.input_cache"], "train.model_mlp_deep_out"),
            # node(predict, ["model_mlp_deep", "test.input_cache"], "test.model_mlp_deep_out"),
            # final dataset
            # node(merge_pressure, ["test.raw", "test.model_mlp_deep_out"], "submission_mlp_deep",),
        ]
    )
