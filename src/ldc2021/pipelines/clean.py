import pandas as pd
from kedro.pipeline import Pipeline, node


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


def transform_cycle(df: pd.DataFrame) -> pd.DataFrame:
    # remove prefix (either "train" or "test")
    prefix_length = len(df["cycle"].iloc[0].split("0")[0])
    df["cycle"] = df["cycle"].map(lambda e: int(e[prefix_length:]))

    return df


def select_label(df: pd.DataFrame) -> pd.DataFrame:
    assert "pa" in df
    return df["pa"].to_frame()


def drop_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(["pa"], axis="columns", errors="raise")
