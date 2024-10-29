from abc import abstractmethod, ABC
from copy import deepcopy
# from autoop.core.ml.artifact import Artifact
import numpy as np


class Model(ABC):
    """Base class for all models used in the assignment."""
    def __init__(self, type: str = None) -> None:
        """Initialize the Model class by creating the artifact"""
        self._parameters: dict = {}
        self._type = type

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

    @property
    def type(self) -> str:
        """ Returns the model type. """
        return self._type
