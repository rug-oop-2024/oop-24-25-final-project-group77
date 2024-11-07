from pydantic import BaseModel, Field
import pickle
import base64
import os


class Artifact(BaseModel):
    """ A class to represent an ML artifact"""
    name: str = Field(title="Name of the asset")
    asset_path: str = Field(title="Path to the asset")
    version: str = Field(title="Version of the asset")
    data: bytes = Field(title="Data of the asset")
    metadata: dict = Field(title="Metadata of the asset", default_factory=dict)
    type: str = Field(title="Type of the asset")
    tags: list = Field(title="Tags of the asset", default_factory=list)

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
        """ Read data from a given path """
        return self.data

    def save(self, data: bytes) -> None:
        """
        Save the artifact's data to the specified asset path.
        Raises an exception if the directory does not exist.
        """
        self.data = data
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        with open(self.asset_path, 'wb') as file:
            pickle.dump(self, file)
