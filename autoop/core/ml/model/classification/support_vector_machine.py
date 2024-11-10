from sklearn.svm import SVC
from typing import Literal, Tuple
from copy import deepcopy

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class SVMClassifier(Model):
    """Wrapper for the Support Vector Machine Classifier"""
    def __init__(self, C: int = 1.0, kernel: Literal['linear', 'poly', 'rbf',
                                                     'sigmoid'] = 'rbf',
                 degree: int = 3, gamma: str = 'scale') -> None:
        """
        Initialize the Support Vector Machine model with various
        hyperparameters, as defined in the scikit-learn library.
        :param C: Inverse of regularization strength
        :param kernel: Type of kernel
        :param degree: Degree of polynomial kernel
        :param gamma: Kernel coefficient
        """
        super().__init__(type="classification")
        C, kernel, degree, gamma = self._validate_hyperparameters(
            C, kernel, degree, gamma)
        self._model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        self._hyperparameters = {"C": C, "kernel": kernel, "degree": degree,
                                 "gamma": gamma}

    def _validate_hyperparameters(
        self,
        C: float,
        kernel: Literal['linear', 'poly', 'rbf', 'sigmoid'],
        degree: int,
        gamma: str
    ) -> Tuple[float, Literal['linear', 'poly', 'rbf', 'sigmoid'], int, str]:
        """
        Validates the parameters for the model.
        Replaces every wrong parameter with its default
        value while informing the user of the change.
        :param C: Inverse of regularization strength
        :param kernel: Type of kernel
        :param degree: Degree of polynomial kernel
        :param gamma: Kernel coefficient
        :returns: validated parameters
        """
        if not isinstance(C, float):
            print("C, the regularization parameter, must be a float. "
                  "Setting to default value 1.0")
            C = 1.0
        if C <= 0:
            print("C, the regularization parameter, must be positive. "
                  "Setting to default value 1.0")
            C = 1.0

        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            print("Kernel must be 'linear', 'poly', 'rbf', or 'sigmoid'. "
                  "Setting to default 'rbf'")
            kernel = 'rbf'

        if not isinstance(degree, int):
            print("Degree must be an integer. "
                  "Setting to default value 3")
            degree = 3
        if degree <= 0:
            print("Degree must be positive. "
                  "Setting to default value 3")
            degree = 3

        if gamma != 'scale':
            print("Gamma must be 'scale'. Setting to default 'scale'.")
            gamma = 'scale'

        return C, kernel, degree, gamma

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the SVM method .fit
        :param observations: data to fit on
        :param ground_truth: labels of the observations
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "_intercept": self._model.intercept_,
            "_support_vectors": self._model.support_vectors_,
            "_dual_coef": self._model.dual_coef_,
            "_classes": self._model.classes_
        }
        if self._model.kernel == 'linear':  # only save coef if linear kernel
            self._parameters["_coef"] = self._model.coef_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the SVM method .predict
        :param observations: data to make predictions on
        :returns: predictions made by the model
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'SVMClassifier':
        """
        Returns a copy of model to prevent leakage. \
        :returns: model
        """
        return deepcopy(self._model)
