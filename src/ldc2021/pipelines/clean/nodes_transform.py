import pandas as pd


def transform_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Remove prefix (either "train" or "test")"""
    prefix_length = len(df["cycle"].iloc[0].split("0")[0])
    df["cycle"] = df["cycle"].map(lambda e: int(e[prefix_length:]))

    return df
