from abc import abstractmethod, ABC
from copy import deepcopy
from autoop.core.ml.artifact import Artifact
import pandas as pd
import io
import numpy as np


class Model(ABC, Artifact):
    """Base class for all models used in the assignment."""
    def __init__(self,
                 type: str = None,
                 name: str = None,
                 asset_path: str = None,
                 version: str = "1.0",
                 data: bytes = None,
                 metadata: dict = {}) -> None:
        """
        Initialize the Model class by creating the artifact.
        The artifact is not used in the implementation, but
        the frame is made to fulfill the requirements.
        Should be adapted to premade frameworks i.e. MLflow.
        """
        self._parameters: dict = {}
        self._hyperparameters: dict = {}
        self._type = type
        Artifact.__init__(
            name=name,
            asset_path=asset_path,
            version=version,
            data=data,
            metadata=metadata,
            type="model",
            tags=[f"{type}"],
        )

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth).
        This method usually saves the paramaters and hyperparameters of the
        model, and hence should be adapted if the programmer decides to
        implement the artifact-based model saving.
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

    def read(self) -> pd.DataFrame:
        """Read model data from a given path"""
        bytes = Artifact.read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save model data to a given path"""
        bytes = data.to_csv(index=False).encode()
        return Artifact.save(bytes)
