from kedro.pipeline import Pipeline, node

from ...utils.columns import drop_features, select_cycle, select_pressure
from .nodes_transform import transform_cycle


def create_pipeline():
    return Pipeline(
        [
            # clean test data (transform, split of cycle)
            node(transform_cycle, ["test.raw"], "test.TMP_clean-test"),
            node(
                drop_features, ["test.TMP_clean-test", "params:drop.test"], "test.clean"
            ),
            node(select_cycle, ["test.TMP_clean-test"], "test.cycle"),
            # clean train data (transform, split of cycle, and pressure)
            node(transform_cycle, ["train.raw"], "train.TMP_clean-train"),
            node(
                drop_features,
                ["train.TMP_clean-train", "params:drop.train"],
                "train.clean",
            ),
            node(select_cycle, ["train.TMP_clean-train"], "train.cycle"),
            node(select_pressure, ["train.TMP_clean-train"], "train.pressure"),
        ]
    )
