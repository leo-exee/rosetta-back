import json

from fastapi import status

from app.models.error_response import ErrorResponse
from app.models.exercise_dto import (
    ExerciseInDTO,
    ExerciseListOutDTO,
)
from app.services.gemini_service import generate_with_gemini_service
from app.utils.prompt_utils import build_prompt


async def generate_exercises_service(input_dto: ExerciseInDTO) -> ExerciseListOutDTO:
    prompt = build_prompt(input_dto)
    raw_response = generate_with_gemini_service(prompt)
    if not raw_response:
        raise ErrorResponse(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to generate exercises.",
            "FAILED_GENERATE_EXERCISES",
        )

    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        json_str = raw_response[start:end]
        json_data = json.loads(json_str)

        return ExerciseListOutDTO(**json_data)

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
