from pathlib import Path
import json
import joblib
import pandas as pd
from typing import Tuple
import push_model
import os


def load_model(model_path: str) -> push_model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist")

    loaded_model = joblib.load(model_path)
    return loaded_model


def handler_predict(event: dict, _) -> dict[str, any]:
    """Predicts output based on GradientBoostingTree model. The input data has the users features and relation with the product information

    Args:
        - event: dictionary with model parameters:
        (
            "user_parameters":
            {
                "user_id1": {"feature 1": value,
                             "feature 2": value,
                             ...
                             },
                "user_id2": {"feature 1": value,
                             "feature 2": value,
                             ...
                             },
                ....
            }
            "model_path": value (extension: .joblib)
        )

    Returns:
        response: Dictionary with "statusCode" (200 if OK; 500 if ERR) and "body" with the "prediction": dict, with the prediction output for each user_id
    """
    model_path = event.get("model_path", None)
    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        return {
            "statusCode": "500",
            "body": json.dumps(
                {
                    "error_message": str(e),
                }
            ),
        }

    data_to_predict = pd.DataFrame.from_dict(json.loads(event["user_parameters"]))
    predictions = model.predict(data_to_predict)

    user_ids = event["user_parameters"].keys()
    user_predictions = dict(zip(user_ids, predictions))
    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": {user_predictions}}),
    }
