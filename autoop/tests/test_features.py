import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from autoop.core.ml.dataset import Dataset  # noqa : E402
from autoop.core.ml.feature import Feature  # noqa : E402
from autoop.functional.feature import detect_feature_types  # noqa : E402

class TestFeatures(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_detect_features_continuous(self):
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")

    def test_detect_features_with_categories(self):
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in numerical_columns,
                                       features):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(lambda x: x.name in categorical_columns,
                                       features):
            self.assertEqual(detected_feature.type, "categorical")


if __name__ == "__main__":
    unittest.main()
