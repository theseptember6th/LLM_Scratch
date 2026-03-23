from pydantic import BaseModel, Field
from typing import Annotated


class PredictionResponse(BaseModel):
    Output: Annotated[str, Field(..., description="The output message from AI")]
