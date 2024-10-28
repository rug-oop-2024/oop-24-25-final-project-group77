from abc import ABC, abstractmethod
import numpy as np

#  get_metrics implemented at the end of the file (!!!)


class Metric(ABC):
    """Base class for all metrics. All metrics must implement _calculate()"""
    _name = None
    _description = None
    _task_type = None

    def __str__(self) -> str:
        """ String representation of any Metric """
        return f"{self._name}: {self._description}" + \
               f", specified for {self._task_type}"

    @abstractmethod
    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the metric """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Allows for calling the metric.
        Calls ValueErrors if y_true and y_pred are not numpy arrays
        of the same size.
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(
                                                    y_pred, np.ndarray):
            raise ValueError("both y_true and y_pred must be numpy arrays")
        if y_true.shape != y_pred.shape:
            print(y_true.shape, y_pred.shape)
            raise ValueError("y_true and y_pred must have the same size")
        return self._calculate(y_true, y_pred)

    @property
    def task_type(self):
        """ Getter for the task type """
        return self._task_type


class MeanSquaredError(Metric):
    """ Implements the Mean Squared Error metric"""
    _name = "Mean Squared Error"
    _description = "Calculates the average of all" + \
        " the squared differences between the ground truth and the prediction"
    _task_type = "regression"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the mean squared error """
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Metric):
    """ Implements the Root Mean Squared Error metric"""
    _name = "Root Mean Squared Error"
    _description = "Calculates the root of the average of all" + \
        " the squared differences between the ground truth and the prediction"
    _task_type = "regression"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the root mean squared error """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class R2(Metric):
    """ Implements the R^2 metric """
    _name = "R^2"
    _description = "Calculates the R^2 score, representing the" + \
        " goodness of fit of a linear model to the data"
    _task_type = "regression"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the R2 score """
        sum_s_res = np.sum((y_true - y_pred) ** 2)
        sum_s_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - sum_s_res / sum_s_tot


class Accuracy(Metric):
    """ Implements the Accuracy metric """
    _name = "Accuracy"
    _description = "Calculates the percentage of correct predictions"
    _task_type = "classification"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the mean accuracy """
        return np.mean(y_true == y_pred)


class Recall(Metric):
    """ Implements the Recall metric """
    _name = "Recall"
    _description = "Calculates the percentage of correct positive" + \
        " predictions when compared to all true positive labels"
    _task_type = "classification"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Apply multiclass recall calculation """
        unique_classes = np.unique(y_true)
        recall_scores, class_instances = [], []

        for cls in unique_classes:
            true_positive = np.sum((y_true == cls) & (y_pred == cls))
            false_negative = np.sum((y_true == cls) & (y_pred != cls))
            total_predictions = true_positive + false_negative

            if total_predictions == 0:
                recall = 0
            else:
                recall = true_positive / total_predictions

            recall_scores.append(recall)
            class_instances.append(np.sum(y_true == cls))

        # Apply weighting (standard way of handling multiclass problems)
        number_of_predictions = len(y_pred)
        weighted_recall = np.sum(np.array(recall_scores) *
                                 np.array(class_instances)
                                 ) / number_of_predictions
        return weighted_recall


class Precision(Metric):
    """ Implements the Precision metric """
    _name = "Precision"
    _description = "Calculates the percentage of correct positive" + \
        " predictions when compared to all predicted positive labels"
    _task_type = "classification"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Apply multiclass precision calculation """
        all_classes = np.unique(y_true)
        precision_scores, class_instances = [], []

        for class_ in all_classes:
            true_positives = np.sum((y_true == class_) & (y_pred == class_))
            false_positives = np.sum((y_true != class_) & (y_pred == class_))
            total_predictions = true_positives + false_positives

            if total_predictions == 0:
                print("Precision is ill-defined for classes with no",
                      "predictions. setting precision to 0")
                precision = 0
            else:
                precision = true_positives / total_predictions

            precision_scores.append(precision)
            class_instances.append(np.sum(y_true == class_))

        # Apply weighting (standard way of handling multiclass problems)
        number_of_predictions = len(y_pred)
        weighted_precision = np.sum(np.array(precision_scores) *
                                    np.array(class_instances)
                                    ) / number_of_predictions
        return weighted_precision


class F1(Metric):
    """ Implements the F1-score metric """
    _name = "F1"
    _description = "Calculates the F1 score, representing the" + \
        " harmonic mean of precision and recall"
    _task_type = "classification"

    def _calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Apply multiclass F1-score calculation """
        helper_class_recall = Recall()
        helper_class_precision = Precision()
        recall = helper_class_recall(y_true, y_pred)
        precision = helper_class_precision(y_true, y_pred)
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return f1


METRICS = [
    "MSE",
    "RMSE",
    "R2",
    "Accuracy",
    "Recall",
    "Precision",
    "F1"
]  # add the names (in strings) of the metrics you implement

METRICS_DICT = {
    "MSE": MeanSquaredError,
    "RMSE": RootMeanSquaredError,
    "R2": R2,
    "Accuracy": Accuracy,
    "Recall": Recall,
    "Precision": Precision,
    "F1": F1,
}  # Mapping from name to class


def get_metric(name: str) -> Metric:
    # Factory function to get a metric by name.
    if name not in METRICS:
        print(f"Metric {name} is not yet implemented.")
        return None
    # create a class instance of the same name and return it
    metric = METRICS_DICT[name]()
    print("Initializing metric- ", metric)
    print("!!!Note that this metric should only be used for",
          f"{metric.task_type} tasks!!!")
    return metric
