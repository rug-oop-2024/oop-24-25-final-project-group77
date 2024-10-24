from sklearn.svm import SVC
from typing import Literal
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
    def __init__(self, C=1.0, kernel: Literal['linear', 'poly', 'rbf',
                                              'sigmoid'] = 'rbf',
                 degree=3, gamma='scale') -> None:
        """
        Initialize the Support Vector Machine model with various
        hyperparameters, as defined in the scikit-learn library.
        :param C: Inverse of regularization strength
        :param kernel: Type of kernel
        :param degree: Degree of polynomial kernel
        :param gamma: Kernel coefficient
        """
        self._model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the SVM method .fit
        """
        self._model.fit(observations, ground_truth)

        self._parameters = {
            "_intercept": self._model.intercept_,
            "_support": self._model.support_,
            "_support_vectors": self._model.support_vectors_,
            "_n_support": self._model.n_support_,
            "_dual_coef": self._model.dual_coef_,
            "_classes": self._model.classes_
        }
        if self._model.kernel == 'linear':  # only save coef if linear kernel
            self._parameters["_coef"] = self._model.coef_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the SVM method .predict
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'SVMClassifier':
        """ Returns a copy of model to prevent leakage. """
        return deepcopy(self._model)


#test the class for functionality on a random sklearn dataset

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.1, random_state=42
    )

    model = SVMClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(predictions)