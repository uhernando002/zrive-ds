import pandas as pd
import os
import re
import numpy as np
from datetime import datetime


DATA_PATH = "/home/unai/datasets/groceries/"
TABLES = {"abandoned_carts": 0, "users": 1, "regulars": 2, "orders": 3, "inventory": 4}


def main():
    a = extract_data(DATA_PATH)
    users_using_regulars(a[TABLES["regulars"]], a[TABLES["users"]])
    items_being_ordered(a[TABLES["inventory"]], a[TABLES["orders"]])
    frequent_item_types(a[TABLES["inventory"]], a[TABLES["orders"]], 2022)


def extract_data(data_path: str):
    data_list = []
    for file_name in os.listdir(data_path):
        data_list.append(
            pd.read_parquet(
                os.path.join(data_path, file_name), engine="pyarrow"
            ).reset_index(drop=True)
        )
        print(file_name)
    print(data_list)
    return data_list
    # print(data.columns.tolist())
    # print(data.head())
    # print(data.info())
    # print(data.groupby(["table"])["user_id"].nunique())
    # print(data["product_type"].nunique())
    # a = data[data["table"] == "inventory"]
    # print(a.groupby(["user_id"])["variant_id"])


def users_using_regulars(
    regulars_info: pd.DataFrame,
    users_info: pd.DataFrame,
):  # HIPOTESIS: Cuantos users usan la opción regulars
    number_of_users = len(users_info)
    percentage = regulars_info.user_id.nunique() * 100 / number_of_users
    print(f"The number of users that create regular items is of {round(percentage,2)}%")


def items_being_ordered(
    inventory_info: pd.DataFrame,
    orders_info: pd.DataFrame,
):  # HIPOTESIS: Cuantos items han sido pedidos: RESULTADOS MUY RAROS !!!
    number_of_items = len(inventory_info)
    ordered_items = (
        orders_info["ordered_items"].explode().reset_index(drop=True).unique()
    )
    percentage = len(ordered_items) * 100 / number_of_items
    print(f"The number of items that have been ordered is of {round(percentage,2)}%")
    print(len(np.intersect1d(ordered_items, inventory_info["variant_id"])))


def frequent_item_types(
    inventory_info: pd.DataFrame,
    orders_info: pd.DataFrame,
    year: int,
):  # HIPOTESIS: Que tipo de productos son más comprados en cada año
    idxs = [
        idx for idx, order in orders_info.iterrows() if order["order_date"].year == year
    ]
    ordered_items = orders_info.iloc[idxs, :]
    ordered_items = orders_info["ordered_items"].explode().reset_index(drop=True)
    print(ordered_items)
    ordered_items_types = pd.Series("NaN", index=np.arange(len(ordered_items)))
    for k, ordered_item in enumerate(ordered_items):
        item_info = inventory_info.loc[inventory_info["variant_id"] == ordered_item]
        if not item_info.empty:
            ordered_items_types[k] = item_info["product_type"].values[0]
    print(ordered_items_types.value_counts().to_dict())


if __name__ == "__main__":
    main()
