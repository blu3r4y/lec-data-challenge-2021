from kedro.pipeline import Pipeline, node

from ...utils.columns import merge_pressure
from ...utils.selection import get_feature_selection_for
from .nodes import predict, train


def create_pipeline():
    ns_train = "train"
    ns_infer = "test"

    # DO NOT namespace this pipeline
    return Pipeline(
        [
            # train on train set
            node(
                get_feature_selection_for(ns_train),
                ["params:selection.minimal"],
                f"{ns_train}.input_cache",
            ),
            node(
                train,
                [f"{ns_train}.input_cache", f"{ns_train}.pressure", "params:ridge"],
                "model_ridge",
            ),
            # infer on test set
            node(
                get_feature_selection_for(ns_infer),
                ["params:selection.minimal"],
                f"{ns_infer}.input_cache",
            ),
            node(
                predict, ["model_ridge", f"{ns_infer}.input_cache"], "model_ridge_out",
            ),
            node(merge_pressure, [f"{ns_infer}.raw", "model_ridge_out"], "submission",),
        ]
    )
