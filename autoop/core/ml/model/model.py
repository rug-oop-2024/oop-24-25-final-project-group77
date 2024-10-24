from abc import abstractmethod, ABC
from copy import deepcopy
#from autoop.core.ml.artifact import Artifact

import numpy as np


class Model(ABC):
    """Base class for all models used in the assignment."""
    def __init__(self):
        self._parameters = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth).
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Make predictions based on the observations."""
        pass

    @property
    def parameters(self) -> dict:
        """ Returns a copy of parameters to prevent leakage. """
        return deepcopy(self._parameters)
