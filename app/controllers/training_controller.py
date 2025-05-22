import asyncio
import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.services.training_service import train_model_service

training_controller = APIRouter(
    prefix="/train",
    tags=["train"],
)


@training_controller.post(
    "/start",
    summary="Start training the model",
    description="This endpoint starts the training process for the model.",
)
async def start_training():
    logging.info("Starting training process...")
    asyncio.create_task(train_model_service())
    return JSONResponse(
        content={"message": "Training process started."},
        status_code=status.HTTP_202_ACCEPTED,
    )
