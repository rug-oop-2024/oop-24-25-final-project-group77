from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """
    Artifact: an abstract object refering to an asset which is stored and
    includes information about this specific asset (e.g., datasets, models,
    pipeline outputs, etc.).

    """
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
