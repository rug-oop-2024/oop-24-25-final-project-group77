import numpy as np
from collections import Counter
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class KNearestNeighbors(Model):
    """ Class that defines K-Nearest Neighbours model."""
    def __init__(self, k: int = 3) -> None:
        """
        Initialize the KNN model with various hyperparameters,
        including the number of nearest neighbours.
        """
        super().__init__(type="classification")
        self.k = self._validate_hyperparameters(k)

    def _validate_hyperparameters(self, v: int) -> int:
        """ Validate k to ensure that it is an int larger than 0. """
        if not isinstance(v, int):
            print('K must be an integer. Setting to default value 3')
            v = 3
        if v < 1:
            print('k must be > 0. Setting to default value 3')
            v = 3
        return v

    @property
    def k(self) -> int:
        """
        Getter for the number of nearest neighbours.
        The getter is used as this model allows dynamic modification
        of its hyperparameters, as they are not used in training.
        """
        return self._k

    @k.setter
    def k(self, v: int) -> None:
        """
        Setter for the number of nearest neighbours.
        This is the only model that allows a hyperparameter to be set by
        the user, as this paramater is not used in training.
        """
        self._k = self._validate_hyperparameters(v)
        self._hyperparameters = {"k": v}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the KNN model by storing the input observations
        and ground truth labels into a dictionary of parameters.
        """
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Generates predictions by iterating through each
        observation and predicting per observation as per
        the method _predict_single. Returns all the predictions.
        """
        predictions = [self._predict_single(x) for x in observations]
        predictions = np.array(predictions, dtype=int)
        return predictions

    def _predict_single(self, observations: np.ndarray) -> int:
        """
        This method implements the KNN algorithm, as described
        in Tutorial 1.  We are not the authors of this algorithm.
        """
        # step1: calc distance between observation and every other point
        dist = np.linalg.norm(self._parameters["observations"] - observations,
                              axis=1)
        # step2: sort the array of the distances and take the first k
        k_indices = np.argsort(dist)[:self.k]
        # step3: check the label aka ground truth of those points
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in
                            k_indices]
        # step4: take most common label and return it to the caller
        return Counter(k_nearest_labels).most_common()[0][0]
