from xgboost import XGBRegressor as WrappedXGBRegressor
from typing import Tuple
from copy import deepcopy

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class XGBRegressor(Model):
    """ XGBoost for regression wrapper """
    def __init__(self,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 gamma: float = 0.0,
                 ) -> None:
        """
        Initialize the XGBoost model with various hyperparameters,
        as defined in the scikit-learn library.
        :param max_depth: Maximum depth
        :param learning_rate: Learning rate
        :param n_estimators: Number of estimators
        :param gamma: Minimum loss reduction
        """
        super().__init__(type="regression")
        max_depth, learning_rate, n_estimators, gamma = \
            self._validate_parameters(max_depth, learning_rate, n_estimators,
                                      gamma)
        self._model = WrappedXGBRegressor(max_depth=max_depth,
                                          learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          gamma=gamma)
        self._hyperparameters = {"max_depth": max_depth,
                                 "learning_rate": learning_rate,
                                 "n_estimators": n_estimators,
                                 "gamma": gamma}

    def _validate_parameters(
        max_depth: int,
        learning_rate: float,
        n_estimators: int,
        gamma: float
    ) -> Tuple[int, float, int, float]:
        """
        Validates the parameters for the model.
        Replaces every wrong parameter with its default
        value while informing the user of the change.
        """
        if not isinstance(max_depth, int):
            print("Max depth must be an integer. Setting to default value 6")
            max_depth = 6
        if not isinstance(learning_rate, float):
            print("Learning rate must be a float. "
                  "Setting to defaul value 0.1")
            learning_rate = 0.1
        if not isinstance(n_estimators, int):
            print("Number of estimators must be an integer. "
                  "Setting to default value 100")
            n_estimators = 100
        if not isinstance(gamma, float):
            print("Minimum loss reduction 'gamma' must be a float. "
                  "Setting to default value 0.0")
            gamma = 0.0

        if learning_rate < 0.0 or learning_rate > 1.0:
            print("Learning rate must be positive and between [0.0, 1.0]. "
                  "Setting to default value 0.1")
            learning_rate = 0.1
        if max_depth < 0:
            print("Max depth must be positive. Setting to default value 6")
            max_depth = 6
        if n_estimators < 0:
            print("Number of estimators must be positive. "
                  "Setting to default value 100")
            n_estimators = 100
        if gamma < 0.0:
            print("Minimum loss reduction 'gamma' must be positive. "
                  "Setting to default value 0.0")
            gamma = 0.0

        return max_depth, learning_rate, n_estimators, gamma

    @property
    def validate_parameters(self) -> int:
        """
        Getter for the validator so that the user can check the allowed range.
        Returns a deepcopy as functions are mutable.
        """
        return deepcopy(self._validate_parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the xgboost method .fit
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "booster": self._model.get_booster(),
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the xgboost method .predict
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'XGBRegressor':
        """ Returns a copy of model to prevent leakage. """
        return deepcopy(self._model)
