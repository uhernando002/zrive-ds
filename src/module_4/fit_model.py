from pathlib import Path
import json
import joblib
from typing import Tuple
from .push_model import PushModel
from .utils import load_clear_data
from datetime import datetime

DEFAULT_THRESHOLD = 0.28


def extract_parameters(event: dict) -> Tuple[dict, float]:
    model_parameters = event["model_parameters"]
    prediction_threshold = event.get("model_threshold", DEFAULT_THRESHOLD)
    return (model_parameters, prediction_threshold)


def save_model(model) -> Path:
    """Saves the trained GradientBoostingTree model on disk with joblib extension

    Args:
        - model: fitted model

    Raises:
        FileExistsError: if the fie path of the model already exists

    Returns:
        file_path = path to the stored model (type: model_name_pushh_yyyy_mm_dd where yyyy_mm_dd is the training date)
    """

    file_path = (
        Path(__file__).parent.parent.parent.parent.resolve()
        / f"models/{create_model_file_name('GradientBoostingTree')}"
    )

    if file_path.exists():
        raise FileExistsError(f"{file_path} alrady exists")

    joblib.dump(model, file_path)
    return file_path


def create_model_file_name(model_name: str) -> str:
    actual_date = datetime.today().strftime("%Y-%m-%d")
    model_name = f"GradientBoostingTree_push_{actual_date}.joblib"
    return model_name


def handler_fit(event: dict, _) -> dict[str, any]:
    """Fits the model with GradientBoostingTree

    Args:
        - event: dictionary with model parameters:
        (
            "model_parameters":
            {
                "GradientBoosting param1": value,
                "GradientBoosting param2": value,
                ....
            }
            "model_threshold": value
        )
        The model_threshold is optional, only model_parameters is mandatory

        Example:
            (
                "model_parameters":
                {
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                }
            )

    Returns:
        response: Dictionary with "statusCode" (200 if OK; 500 if ERR) and "body" with the "model_path": value, where the model has been saved
    """

    model_parameters, prediction_threshold = extract_parameters(event)
    model = PushModel(model_parameters, prediction_threshold)

    trainnig_data = load_clear_data()
    model.fit(trainnig_data)
    try:
        model_path = save_model(model)
    except FileExistsError as e:
        return {
            "statusCode": "500",
            "body": json.dumps(
                {
                    "error_message": str(e),
                }
            ),
        }

    return {
        "statusCode": "200",
        "body": json.dumps(
            {
                "model_path": [model_path],
            }
        ),
    }
