from xgboost import XGBRegressor as WrappedXGBRegressor
from pydantic import PrivateAttr
from typing import Literal
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
                 objective: Literal['reg:squarederror', 'reg:absoluteerror',
                                    'reg:squaredlogerror'
                                    ] = 'reg:squarederror',
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 gamma: float = 0.0,
                 reg_lambda: float = 1.0,
                 reg_alpha: float = 0.0
                 ) -> None:
        """
        Initialize the XGBoost model with various hyperparameters,
        as defined in the scikit-learn library.
        :param objective: Objective function
        :param max_depth: Maximum depth
        :param learning_rate: Learning rate
        :param n_estimators: Number of estimators
        :param subsample: Subsample ratio
        :param colsample_bytree: Subsample ratio of columns
        :param gamma: Minimum loss reduction
        :param reg_lambda: L2 regularization
        :param reg_alpha: L1 regularization
        """
        self._model = WrappedXGBRegressor(objective=objective,
                                          max_depth=max_depth,
                                          learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          subsample=subsample,
                                          colsample_bytree=colsample_bytree,
                                          gamma=gamma, reg_lambda=reg_lambda,
                                          reg_alpha=reg_alpha)

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
