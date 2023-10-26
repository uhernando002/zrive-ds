import pandas as pd
from pathlib import Path


def load_all_data() -> pd.DataFrame:
    data_file = (
        Path(__file__).parent.parent.parent.parent.resolve()
        / "datasets/feature_frame.csv"
    )

    data = pd.read_csv(data_file)


def preprocess_data(df: pd.DataFrame, maxItems: int = 5) -> pd.DataFrame:
    # We only keep orders with at least 5 items
    orders = df["order_id"].unique()
    interested_orders = orders[df.groupby("order_id")["outcome"].sum() >= maxItems]
    return df.loc[df["order_id"].isin(interested_orders)]


def load_clear_data() -> pd.DataFrame:
    raw_data = load_all_data()
    clear_data = preprocess_data(raw_data)
    return clear_data
