import pandas as pd


def select_label(df: pd.DataFrame) -> pd.DataFrame:
    assert "pa" in df
    return df["pa"].to_frame()


def drop_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(["pa"], axis="columns", errors="raise")
