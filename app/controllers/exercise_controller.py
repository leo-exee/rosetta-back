from fastapi import APIRouter

from app.services.exercise_service import (
    ExerciseRequest,
    exercise_service,
)

exercise_controller = APIRouter(
    prefix="/exercises",
    tags=["exercises"],
)


@exercise_controller.post("/generate")
async def generate_exercises(request: ExerciseRequest):
    exercises = exercise_service.generate_exercises(request)
    return exercises
