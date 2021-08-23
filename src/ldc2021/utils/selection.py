# Copyright 2021
#    Dynatrace Research
#    SAL Silicon Austria Labs
#    LIT Artificial Intelligence Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from typing import Callable, Dict, Iterable, List, Union

import pandas as pd
from funcy import first
from kedro.framework.session import get_current_session
from kedro.pipeline import Pipeline
from kedro.pipeline.node import node

log = logging.getLogger(__name__)

SelectionType = List[Union[str, Dict[str, List[str]]]]


def get_feature_selection_for_all(prefixes: Iterable[str], param_key: str) -> Pipeline:
    """
    Just a helper that gives you a small pipeline to compute features for all prefixes
    :param prefixes: The prefixes for which this feature selection shall be done
    :param param_key: The name of the parameter without the "params:" prefix
    :return: a pipeline holding the necessary nodes
    """
    return Pipeline(
        [
            node(
                get_feature_selection_for(prefix),
                [f"params:{param_key}"],
                f"{prefix}.s05_features",
            )
            for prefix in prefixes
        ]
    )


def get_feature_selection_for(prefix: str) -> Callable[[SelectionType], pd.DataFrame]:
    """
    Just a helper that will give a callable instance of `get_feature_selection` with
    an already set `prefix` for easy usage inside kedro pipelines
    :param prefix: Will load the entries from the data catalog with this prefix
    :return: The function `get_feature_selection` without a pre-filled `prefix` argument
    """
    fun = partial(get_feature_selection, prefix=prefix)
    fun.__name__ = f"get_feature_selection(prefix={prefix})"  # for kedro logs
    return fun


def get_feature_selection(selection: SelectionType, prefix: str = None) -> pd.DataFrame:
    """
    Creates a single dataframe that holds all the specified features by
    directly querying it from the data catalog with a specified `prefix`.

    The `selection` can be read from YAML and must be specified like in the example below.
    This will combine the three datasets `s03_selection`, `s03_log` and `s04_ip_parts`,
    although, from the last one it will only take the three columns `dst_3`, `dst_2`, `dst_1`

    >>> sel = ["s03_selection", "s03_log", { "s04_ip_parts": ["dst_3", "dst_2", "dst_1"] }]

    :param selection: A list of features from the data catalog, without prefixes.
        If the list holds a dictionary, its one and only key is used as the name of the feature
        and its value shall be a list that holds the explicit columns to add from this catalog entry
    :param prefix: Will load the entries from the data catalog with this prefix
    :return: A single dataframe that holds all the selected features
    """

    session = get_current_session()
    context = session.load_context()
    catalog = context.catalog
    frames = {}

    for entry in selection:

        if isinstance(entry, str):
            feature_df = catalog.load(f"{prefix}.{entry}")
            frames[entry] = feature_df

        elif isinstance(entry, dict):
            assert len(entry) == 1  # select the one and only (key, value) pair
            name, columns = first(entry.items())

            # additionally perform a specific column selection
            feature_df = catalog.load(f"{prefix}.{name}")
            feature_df = feature_df.loc[:, columns]
            frames[name] = feature_df

        else:
            raise ValueError("invalid type in selection. expected str or dict.")

    result = pd.concat(frames.values(), axis="columns", verify_integrity=True)

    memory = f"{result.memory_usage().sum() / (1024 ** 3):.2f} GB"
    log.info(
        f"selected {result.shape} elements ({memory}) from prefix {prefix} with features {list(frames.keys())}"
    )
    log.info(f"selected columns: {list(result.columns)}")

    return result
