from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """ A class used to represent features of a dataset.  """
    name: str = Field(title="Name of the feature", default=None)
    type: Literal["categorical", "numerical"] = Field(
        title="Type of the feature", default=None
    )

    def __str__(self) -> str:
        """ Return a string representation of the object."""
        return f"Feature name={self.name}, type={self.type}"
