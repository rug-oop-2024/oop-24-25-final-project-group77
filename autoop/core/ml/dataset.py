from autoop.core.ml.artifact import Artifact
# from abc import ABC, abstractmethod
import pandas as pd
import io


class Dataset(Artifact):
    """A class to represent an ML dataset"""

    @property
    def name(self):
        # Optionally, you could add some logic here
        return self._name  # Ensure this matches the internal attribute name

    def __init__(
        self,
        name: str,
        asset_path: str,
        version: str,
        data: bytes,
        metadata: dict = {},
        tags: list = [],
    ):
        super().__init__(
            name=name,
            asset_path=asset_path,
            version=version,
            data=data,
            metadata=metadata,
            type="dataset",
            tags=tags,
        )

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ):
        """Create a dataset from a pandas dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Read data from a given path"""
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save data to a given path"""
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
