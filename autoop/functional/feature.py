from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    ds = dataset.read()
    features = []
    for name in ds.columns:
        if ds[name].dtype == "object":
            features.append(Feature(name=name, type="categorical"))
        else:
            features.append(Feature(name=name, type="numerical"))
    return features
