from fastapi import APIRouter

training_controller = APIRouter(
    prefix="/train",
    tags=["train"],
)
