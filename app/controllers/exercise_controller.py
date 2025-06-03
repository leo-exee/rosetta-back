from fastapi import APIRouter, status

from app.models.error_response import ErrorResponseModel
from app.models.exercise_dto import ExerciseInDTO, ExerciseListOutDTO
from app.services.exercice_service import generate_exercises_service

exercise_router = APIRouter(
    prefix="/exercises",
    tags=["Exercises"],
)


@exercise_router.post(
    "",
    name="generate_exercises",
    summary="Generate Exercises",
    description="Generate exercises based on the provided context and level.",
    response_description="List of generated exercises",
    status_code=status.HTTP_200_OK,
    response_model=ExerciseListOutDTO,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
            "model": ErrorResponseModel,
        },
    },
)
async def generate_exercises_controller(req: ExerciseInDTO) -> ExerciseListOutDTO:
    return await generate_exercises_service(req)
