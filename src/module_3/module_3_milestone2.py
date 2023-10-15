import pandas as pd
import os
import re
import numpy as np
from datetime import datetime


DATA_PATH = "/home/unai/datasets/feature_frame.csv"


def main():
    data = load_data(DATA_PATH)


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


if __name__ == "__main__":
    main()
