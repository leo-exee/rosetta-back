from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: int
    status: str
    message: str
    errors: list
