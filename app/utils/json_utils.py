import json

from fastapi import status

from app.models.error_response import ErrorResponse


def parse_raw_json(raw_json: str) -> dict:
    try:
        start = raw_json.find("{")
        end = raw_json.rfind("}") + 1
        json_str = raw_json[start:end]
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        raise ErrorResponse(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Invalid JSON response: {e}",
            "INVALID_JSON_RESPONSE",
        ) from e
    except Exception as e:
        raise ErrorResponse(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"An unexpected error occurred: {e}",
            "UNEXPECTED_ERROR",
        ) from e
