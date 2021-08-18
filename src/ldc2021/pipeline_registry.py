from functools import reduce
from operator import add
from typing import Dict, Iterable

from kedro.pipeline import Pipeline, pipeline

from .pipelines import clean, features
from .utils.namespaces import NAMESPACES


def register_pipelines() -> Dict[str, Pipeline]:
    def _for_all(pipe: Pipeline, namespaces: Iterable[str] = NAMESPACES):
        """make a single pipeline to a pipeline for all namespaces"""
        return reduce(add, (pipeline(pipe, namespace=ns) for ns in namespaces))

    pp_clean = clean.create_pipeline()
    pp_features = _for_all(features.create_pipeline())

    pp_all = pipeline(pp_clean + pp_features)

    return {
        "clean": pp_clean,
        "features": pp_features,
        "__default__": pp_all,
    }
