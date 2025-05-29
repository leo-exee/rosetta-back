from fastapi import status

from app.models.error_response import ErrorResponse
from app.models.exercise_dto import (
    ExerciseInDTO,
    ExerciseListOutDTO,
)
from app.services.gemini_service import generate_gemini_content_service
from app.utils.json_utils import parse_raw_json
from app.utils.prompt_utils import build_prompt


async def generate_exercises_service(input_dto: ExerciseInDTO) -> ExerciseListOutDTO:
    prompt = build_prompt(input_dto)
    raw_response = generate_gemini_content_service(prompt)
    if not raw_response:
        raise ErrorResponse(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to generate exercises.",
            "FAILED_GENERATE_EXERCISES",
        )

    json_data = parse_raw_json(raw_response)
    return ExerciseListOutDTO(**json_data)
