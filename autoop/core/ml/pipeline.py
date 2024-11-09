from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """ The class representing the pipeline"""
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8
    ) -> None:
        """ Initialize the pipeline """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and model.type \
                != "classification":
            tempstr = "Model type must be classification for categorical"
            tempstr += " target feature"
            raise ValueError(tempstr)
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        input_features_str = "\n    - ".join(map(str, self._input_features))
        metrics_str = "\n    - ".join(map(str, self._metrics))

        return (
            f"Pipeline:\n"
            f"  - Model Type: {self._model.type}\n"
            f"  - Input Features:\n    - {input_features_str}\n"
            f"  - Target {self._target_feature}\n"
            f"  - Train/Test Split: {self._split}\n"
            f"  - Metrics:\n    - {metrics_str}"
        )

    @property
    def model(self) -> Model:
        """
        Returns the given model.
        Isn't copied as was implemented before.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline
        execution to be saved.
        :return: List of artifacts generated during the pipeline execution
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Register an artifact in the pipeline.
        :param name: name of the artifact
        :param artifact: artifact to be registered
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """ Preprocess the features in the pipeline. """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the I/O vectors, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self) -> None:
        """ Split the data into training and testing sets. """
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[: int(
            split * len(self._output_vector))]
        self._test_y = self._output_vector[int(
            split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compact the vectors into a single array.
        :param vectors: list of vectors
        :return: compacted vector
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """ Train the model. """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """ Evaluate the model. """
        X_test = self._compact_vectors(self._test_X)
        X_train = self._compact_vectors(self._train_X)
        Y_test = self._test_y
        Y_train = self._train_y
        self._metrics_results = []
        predictions = self._model.predict(X_test)
        prediction_training = self._model.predict(X_train)
        for metric in self._metrics:
            result_test = metric(predictions, Y_test)
            result_train = metric(prediction_training, Y_train)
            self._metrics_results.append(("training:", metric,
                                          result_train, "test:",
                                          metric, result_test))
        self._predictions = predictions

    def execute(self) -> dict:
        """ Execute the pipeline. """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }

    @staticmethod
    def from_pipeline(
        pipeline: "Pipeline", name: str, version: str, asset_path: str, 
        serialized_pipeline: bytes
         ) -> Artifact:
        """
        Create an Artifact from a Pipeline instance.
        :param pipeline: the pipeline instance to be serialized and saved
        :param name: name of the pipeline artifact
        :param version: version of the pipeline artifact
        :param asset_path: path to save the pipeline artifact
        :return: an Artifact instance representing the serialized pipeline
        """

        # Create and return an Artifact instance
        return Artifact(
            name=name,
            version=version,
            asset_path=asset_path,
            data=serialized_pipeline,
            type="pipeline",
        )
