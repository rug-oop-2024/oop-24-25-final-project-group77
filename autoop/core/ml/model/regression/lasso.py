import numpy as np
from sklearn.linear_model import Lasso as WrappedLasso
from copy import deepcopy

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class Lasso(Model):
    """ Lasso regression model wrapper. """
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize the Lasso model with various hyperparameters,
        as defined in the scikit-learn library.
        :param alpha: Regularization strength
        """
        super().__init__(type="regression")
        alpha = self._validate_alpha(alpha)
        self._model = WrappedLasso(alpha=alpha)
        self._hyperparameters = {"alpha": alpha}

    def _validate_alpha(self, alpha: float) -> float:
        """
        Validate the regularization strength
        """
        if alpha < 0.0:
            print("Regularization strength must be positive.",
                  "Setting to default value 1.0")
        return max(0, alpha)

    @property
    def validate_alpha(self) -> int:
        """
        Getter for the validator so that the user can check the allowed range.
        Returns a deepcopy as functions are mutable.
        """
        return deepcopy(self._validate_alpha)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the Lasso method .fit
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "_coef": self._model.coef_,
            "_intercept": self._model.intercept_
        }  # Splitting the vector into weights and bias

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the Lasso method .predict
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'Lasso':
        """ Returns a copy of model to prevent leakage. """
        return deepcopy(self._model)
