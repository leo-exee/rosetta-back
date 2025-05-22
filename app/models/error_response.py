from typing import Union

from fastapi import status


class ErrorResponse(Exception):
    def __init__(
        self,
        code: str | int,
        message: str,
        status: str,
        details: list[object] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status = status
        self.details = details if details is not None else []

    @classmethod
    def from_error(cls, e: Union["ErrorResponse", Exception]) -> "ErrorResponse":
        if isinstance(e, cls):
            return e
        return cls(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal Server Error",
            "INTERNAL_SERVER_ERROR",
        )
