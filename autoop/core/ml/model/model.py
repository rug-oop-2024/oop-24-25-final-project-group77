from abc import abstractmethod, ABC
from copy import deepcopy
from autoop.core.ml.artifact import Artifact
import numpy as np


class Model(ABC):
    """ Base class for all models used in the assignment. """
    def __init__(self, type: str = None) -> None:
        """
        Initialize the Model class by creating the artifact.
        :param type: type of the model
        """
        self._parameters: dict = {}
        self._hyperparameters: dict = {}
        self._type = type

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth).
        :param observations: data to fit on
        :param ground_truth: labels of the observations
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations.
        :param observations: data to make predictions on
        :returns: predictions made by the model
        """
        pass

    @abstractmethod
    def _validate_hyperparameters(self) -> None:
        """ Validate parameter values passed by the user """
        pass

    def to_artifact(self, name: str) -> Artifact:
        """
        Convert the model to an artifact.
        :param name: name of the artifact
        :returns: Artifact
        """
        return Artifact(name=name, type="model",
                        metadata={"parameters": self.parameters,
                                  "hyperparameters": self.hyperparameters})

    @property
    def validate_hyperparameters(self) -> int:
        """
        Getter for the validator so that the user can check the allowed range.
        Returns a deepcopy as functions are mutable.
        :returns: validator
        """
        return deepcopy(self._validate_hyperparameters)

    @property
    def parameters(self) -> dict:
        """ Returns a copy of parameters to prevent leakage. """
        return deepcopy(self._parameters)

    @property
    def hyperparameters(self) -> dict:
        """ Returns a copy of hyperparameters to prevent leakage. """
        return deepcopy(self._hyperparameters)

    @property
    def type(self) -> str:
        """ Returns the model type. """
        return self._type
