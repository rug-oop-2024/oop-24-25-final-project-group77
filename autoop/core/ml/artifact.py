from pydantic import BaseModel, Field
import pickle
import base64
import os


class Artifact(BaseModel):
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
        """
        base64_asset_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{base64_asset_path}:{self.version}"
    # what the fuck is that class even doing XDDD
    # to figure out: how to know which path to read/save to
    # how this should generally work to ensure anything is saved in the right place
    # and with the right type 
    # how can this work as is done in the pipeline.py and dataset.py
    # and wtf is that lol just use csv files or a proper dataset

    def read(self) -> bytes:
        """ Read data from a given path """
        return self.data

    def save(self) -> None:
        """
        Save the artifact's data to the specified asset path.
        Raises an exception if the directory does not exist.
        """
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)


    @classmethod
    def from_serializable(cls, name: str, data: object, version: str = "1.0.0", 
                          tags: list = None) -> 'Artifact':
        """ Create an artifact from a serializable object """
        serialized_data = pickle.dumps(data)
        return cls(asset_path=name, data=serialized_data, version=version, 
                   tags=tags or [], type="generic")

    def to_serializable(self) -> object:
        """ Convert artifact data back to a serializable object """
        return pickle.loads(self.data)
