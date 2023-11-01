import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib


DATA_PATH = "/home/unai/datasets/feature_frame.csv"
training_columns = [
    "abandoned_before",
    "ordered_before",
    "global_popularity",
    "user_order_seq",
    "normalised_price",
]
label_col = "outcome"
model_filename = "groceries_model.pkl"


def main():
    data = load_data(DATA_PATH)
    # processed_data = preprocessing(data)
    # model = training_model(processed_data)
    # save_model(model, model_filename)
    model = load_model(model_filename)
    threshold = threshold_calculation(data, model)
    # with these 3 variables we can predict new values with the model_prediction function


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # We only keep orders with at least 5 items
    orders = data["order_id"].unique()
    orders_more5 = orders[data.groupby("order_id")["outcome"].sum() >= 5]
    return data.loc[data["order_id"].isin(orders_more5)]


def training_model(data: pd.DataFrame) -> make_pipeline:
    model = make_pipeline(
        StandardScaler(),
        linear_model.LogisticRegression(penalty=None, max_iter=250, solver="lbfgs"),
    )

    return model.fit(data[training_columns], data[label_col])


def save_model(model: make_pipeline, file_name: str):
    joblib.dump(model, file_name)  # file_name must be .pkl


def load_model(model_filename: str) -> make_pipeline:
    try:
        loaded_model = joblib.load(model_filename)
        return loaded_model
    except FileNotFoundError:
        print(f"The model file '{model_filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return None


def threshold_calculation(data: pd.DataFrame, model: make_pipeline) -> np.float64:
    precision, recall, thresholds_precrec = metrics.precision_recall_curve(
        data[label_col], calculate_probabilities(data, model)
    )
    precision_recall = pd.DataFrame(
        {
            "Precision": precision[: len(precision) - 1],
            "Recall": recall[: len(recall) - 1],
            "Threshold": thresholds_precrec,
        }
    )
    return round(
        np.mean(
            precision_recall["Threshold"][
                (precision_recall["Recall"] >= 0.25)
                & (precision_recall["Recall"] <= 0.28)
            ]
        ),
        4,
    )


def calculate_probabilities(data: pd.DataFrame, model: make_pipeline) -> np.array:
    return model.predict_proba(data[training_columns])[:, 1]


def model_prediction(
    data: pd.DataFrame, model: make_pipeline, threshold: np.float64
) -> np.array:
    probabilities = calculate_probabilities(data, model)
    predictions = np.zeros(probabilities.shape)
    predictions[probabilities > threshold] = 1
    return predictions


if __name__ == "__main__":
    main()
