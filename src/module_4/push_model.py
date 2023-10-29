from sklearn import ensemble
import pandas as pd
import numpy as np


class PushModel:
    VARIABLE_COLUMNS = [
        "user_order_seq",
        "ordered_before",
        "abandoned_before",
        "active_snoozed",
        "set_as_regular",
        "normalised_price",
        "discount_pct",
        "global_popularity",
        "count_adults",
        "count_children",
        "count_babies",
        "count_pets",
        "people_ex_baby",
        "days_since_purchase_variant_id",
        "avg_days_to_buy_variant_id",
        "std_days_to_buy_variant_id",
        "days_since_purchase_product_type",
        "avg_days_to_buy_product_type",
        "std_days_to_buy_product_type",
    ]

    LABEL_COLUMN = "outcome"

    def __init__(self, parametrisation: dict, threshold: float) -> None:
        """Initializes the variable with the selected model: GradientBoostingTree

        Args:
            - "parametrisation":
                Dictionary with the values of the parameters of the model. The keys must coincide with the parameter of the model
            - "threshold":
                The value that determines whether the class predicted based on the probability received

        """

        self.model = ensemble.GradientBoostingClassifier(**parametrisation)
        self.thershold = threshold

    def fit(self, data: pd.DataFrame) -> None:
        """It fits the model

        Args:
            - "data":
                Training dataset with variables and the label

        """
        self.model.fit(data[self.VARIABLE_COLUMNS], data[self.LABEL_COLUMN])

    def predict(self, data: pd.DataFrame) -> np.array:
        """It predicts the outcome from the predict_proba probabilities from the value of the variables

        Args:
            - "data":
                Predictable dataset with variables

        Returns:
            - "prediction":
                pd.Series with clasificated predictions for the input data
        """
        probs = self.model.predict_proba(data[self.VARIABLE_COLUMNS])[:, 1]
        predictions = np.zeros(probs.shape)
        predictions[probs > self.thershold] = 1
        return predictions
