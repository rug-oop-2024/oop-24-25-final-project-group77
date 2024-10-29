import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class MultipleLinearRegression(Model):
    """
    Class developed specifically for multiple linear regression, as
    specified in Question 2.
    """
    def __init__(self) -> None:
        """ Initialize the model. """
        super().__init__(type="regression",
                         name="MultipleLinearRegression")

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the formula provided in the assignment.

        :param observations: a 2D ndarray of observations,
        :param ground_truth: a 1D ndarray of ground truth values (labels).
        """
        x_tilde = np.concatenate(
            (np.ones((observations.shape[0], 1)), observations), axis=1
        )
        x_transformed = np.dot(
            np.linalg.inv(np.dot(np.transpose(x_tilde), x_tilde)),
            np.transpose(x_tilde)
        )  # (X^T*X)^-1*X^T
        # Applying formula (11)
        all_parameters = np.dot(x_transformed, ground_truth)
        self._parameters = {
            "_coef": all_parameters[1:],
            "_intercept": all_parameters[0]
        }  # Splitting the vector into weights and biasc

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the output for a given set of observations.

        :param observations: a 2D ndarray of observations,
        :returns: an np.ndarray, consisting of values predicted by the model,
        :raise Exception: if model is not trained or the number of,
        features in the observation is incorrect.
        """
        if not self._parameters:
            raise Exception("Model not trained")
        if len(self._parameters["_coef"]) != len(observations[0]):
            raise Exception("Incorrect number of features")
        coefficients_result = np.dot(observations, self._parameters["_coef"])
        return coefficients_result + self._parameters["_intercept"]
