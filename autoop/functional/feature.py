from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> list[Feature]:
    """
    Detects feature type of a dataset.
    Bases on the assumption that all features are either
    categorical or numerical, and no empty features are present.
    :param dataset: Dataset object, storing the data
    :return: List of features
    """
    ds = dataset.read()
    features = []
    for name in ds.columns:
        if ds[name].dtype == "object":
            features.append(Feature(name=name, type="categorical"))
        else:
            features.append(Feature(name=name, type="numerical"))
    return features
