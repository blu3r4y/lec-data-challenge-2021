from kedro.pipeline import Pipeline, node

from ...utils.columns import drop_label, select_label
from .nodes_transform import transform_cycle


def create_pipeline():
    return Pipeline(
        [
            # clean data frames
            node(transform_cycle, ["test.raw"], "test.clean"),
            node(transform_cycle, ["train.raw"], "train.tmp-clean-1"),
            # split data and labels (for the train data)
            node(drop_label, ["train.tmp-clean-1"], "train.clean"),
            node(select_label, ["train.tmp-clean-1"], "train.pressure"),
        ]
    )
