from fastapi import APIRouter

exercise_controller = APIRouter(
    prefix="/exercises",
    tags=["batch-exercises"],
)
