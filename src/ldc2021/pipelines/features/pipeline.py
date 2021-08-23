from kedro.pipeline import Pipeline, node

from ldc2021.utils.columns import combine_features

from .nodes_math import log_acceleration, sqrt_acceleration
from .nodes_rolling import compute_rolling_std
from .nodes_velocity import compute_velocities


def create_pipeline():
    return Pipeline(
        [
            node(log_acceleration, ["clean"], "log"),
            node(sqrt_acceleration, ["clean"], "sqrt"),
            node(combine_features, ["cycle", "clean"], "cycle_clean"),
            node(
                compute_velocities,
                ["cycle_clean", "params:velocity.absolute"],
                "velocity",
            ),
            node(
                compute_rolling_std,
                ["cycle_clean", "params:rolling.sizes", "params:rolling.type"],
                "rolling",
            ),
        ]
    )
