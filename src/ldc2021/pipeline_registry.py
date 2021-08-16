from typing import Dict

from kedro.pipeline import Pipeline

from ldc2021.pipelines import preprocessing


def register_pipelines() -> Dict[str, Pipeline]:
    preprocessing_pipeline = preprocessing.create_pipeline()

    return {"preprocessing": preprocessing_pipeline, "__default__": Pipeline([])}
