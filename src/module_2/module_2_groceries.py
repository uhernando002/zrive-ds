import pandas as pd
import os
import re


DATA_PATH = "/home/unai/datasets/groceries/"


def main():
    a = extract_data(DATA_PATH)
    join_data(a)


def extract_data(data_path: str):
    data_list = []
    for file_name in os.listdir(data_path):
        data_list.append(
            pd.read_parquet(os.path.join(data_path, file_name), engine="pyarrow")
            .reset_index(drop=True)
            .assign(table=re.sub(".parquet", "", file_name))
        )
        print(file_name)
    print(data_list)
    data = pd.concat(data_list, axis=0, sort=False)
    return data
    # print(data.columns.tolist())
    # print(data.head())
    # print(data.info())
    # print(data.groupby(["table"])["user_id"].nunique())
    # print(data["product_type"].nunique())
    # a = data[data["table"] == "inventory"]
    # print(a.groupby(["user_id"])["variant_id"])


def join_data(extracted_data: pd.DataFrame):
    user_info = extracted_data[extracted_data["table"] == "orders"]
    user_info = user_info.dropna(axis=1, how="all")
    print(user_info.info())


if __name__ == "__main__":
    main()
