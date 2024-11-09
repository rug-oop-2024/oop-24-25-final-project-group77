from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """ A class to represent an ML dataset """

    @property
    def name(self) -> str:
        """ Returns the name of the dataset """
        return self.name

    def __init__(
        self,
        name: str,
        asset_path: str,
        version: str,
        data: bytes,
        metadata: dict = {},
        tags: list = [],
    ) -> None:
        """ Initialize the Dataset class by creating the artifact """
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
    ) -> "Dataset":
        """
        Create a dataset from a pandas dataframe.
        :param data: pandas dataframe containing the data
        :param name: name of the dataset
        :param asset_path: path to the dataset
        :param version: version of the dataset specified by the user
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """ Returns the data read from a given path """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save data to a given path
        :param data: data to be saved
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
