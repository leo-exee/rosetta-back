from fastapi import APIRouter

from app.models.exercise_dto import ExerciseInDTO, ExerciseListOutDTO
from app.services.exercice_service import generate_exercises_service

exercise_router = APIRouter(
    prefix="/exercises",
    tags=["batch-exercises"],
)


@exercise_router.post(
    "",
    summary="Generate Exercises",
    description="Generate exercises based on the provided context and level.",
    response_model=ExerciseListOutDTO,
)
async def fill_in_blank(req: ExerciseInDTO) -> ExerciseListOutDTO:
    return await generate_exercises_service(req)
