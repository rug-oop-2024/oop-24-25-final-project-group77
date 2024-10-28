from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import Lasso
from autoop.core.ml.model.regression import XGBRegressor
from autoop.core.ml.model.classification import MultipleLogisticRegressor
from autoop.core.ml.model.classification import SVMClassifier
from autoop.core.ml.model.classification import KNearestNeighbors


REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "XGBRegressor",
]

CLASSIFICATION_MODELS = [
    "KNearestNeighbors",
    "SVMClassifier",
    "MultipleLogisticRegressor",
]

REGRESSION_MODELS_DICT = {
    "MultipleLinearRegression": MultipleLinearRegression,
    "Lasso": Lasso,
    "XGBRegressor": XGBRegressor,
}

CLASSIFICATION_MODELS_DICT = {
    "KNearestNeighbors": KNearestNeighbors,
    "SVMClassifier": SVMClassifier,
    "MultipleLogisticRegressor": MultipleLogisticRegressor,
}


def get_model(model_name: str) -> Model:
    """ Factory method to get a model by name. """
    if model_name not in REGRESSION_MODELS + CLASSIFICATION_MODELS:
        print(f"Model {model_name} is not yet implemented.")
        return None
    # create a class instance of the same name and return it
    if model_name in REGRESSION_MODELS:
        model = REGRESSION_MODELS_DICT[model_name]
    else:
        model = CLASSIFICATION_MODELS_DICT[model_name]
    return model
