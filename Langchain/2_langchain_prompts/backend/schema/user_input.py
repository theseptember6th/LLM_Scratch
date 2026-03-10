from pydantic import BaseModel, Field, field_validator
from typing import Annotated


class UserInput(BaseModel):
    message: Annotated[str, Field(..., description="Message you want to ask to the AI")]

    @field_validator("message")
    @classmethod
    def normailze_message(cls, value: str) -> str:
        return value.strip().title()
