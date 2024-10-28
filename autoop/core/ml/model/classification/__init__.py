"""
This package contains all classification models.
"""

from autoop.core.ml.model.classification.multiple_logistic_regressor \
    import MultipleLogisticRegressor
from autoop.core.ml.model.classification.support_vector_machine \
    import SVMClassifier
from autoop.core.ml.model.classification.k_nearest_neighbours \
    import KNearestNeighbors


CLASSIFCATION_MODELS_DICT = {
    "MultipleLogisticRegressor": MultipleLogisticRegressor,
    "SVMClassifier": SVMClassifier,
    "KNearestNeighbors": KNearestNeighbors
}  # added to pass style checks while being able to access models
