from kedro.pipeline import Pipeline, node

from .nodes import compute_velocity, transform_cycle


def create_pipeline():
    return Pipeline(
        [
            # train
            node(transform_cycle, ["ldc_train"], "__ldc_train_1"),
            # test
            node(transform_cycle, ["ldc_test"], "__ldc_test_1"),
        ]
    )
