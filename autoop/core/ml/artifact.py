import pickle
import base64
import os


class Artifact():
    """ A class to represent an ML artifact"""
    def __init__(self, name: str, asset_path: str = "", version: str = "",
                 data: bytes = b"", metadata: dict = {}, type: str = "",
                 tags: list = []) -> None:
        """
        Initialize the artifact
        Args:
            name (str): The name of the artifact
            asset_path (str): The path to the artifact
            version (str): The version of the artifact
            data (bytes): The data of the artifact
            metadata (dict): The metadata of the artifact
            type (str): The type of the artifact
        """
        self._name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.metadata = metadata
        self.type = type
        self.tags = tags

    @property
    def name(self) -> str:
        """ Get the name of the artifact """
        return self._name

    @property
    def id(self) -> str:
        """
        Get the id of the artifact
        :returns: str: The id of the artifact
        we use '-' instead of ':' to avoid issues with windows
        """
        base64_asset_path = base64.b64encode(self.asset_path.encode()).decode()
        # REMOVING THE DOUBLE ==, ANOTHER WINDOWS ISSUE
        base64_asset_path = base64_asset_path[:-2]
        return f"{base64_asset_path}-{self.version}"

    def read(self) -> bytes:
        """ Returns data as bytes stored in the artifact"""
        return self.data

    def save(self, data: bytes) -> None:
        """
        Save the artifact's data to the specified asset path.
        Args:
            data (bytes): The data to be saved
        """
        self.data = data
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        with open(self.asset_path, 'wb') as file:
            pickle.dump(self, file)
