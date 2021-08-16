from kedro.pipeline import Pipeline, node

from .nodes import compute_velocity, transform_cycle


def create_pipeline():
    return Pipeline(
        [
            # train
            node(transform_cycle, ["ldc_train"], "__ldc_train_1"),
            node(compute_velocity, ["__ldc_train_1"], "ldc_train_clean"),
            # test
            node(transform_cycle, ["ldc_test"], "__ldc_test_1"),
            node(compute_velocity, ["__ldc_test_1"], "ldc_test_clean"),
        ]
    )
