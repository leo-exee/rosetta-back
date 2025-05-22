import asyncio
import logging

from fastapi import APIRouter

from app.services.training_service import train_service

train_controller = APIRouter(
    prefix="/train",
    tags=["train"],
)


@train_controller.post(
    "/start",
    summary="Start training the model",
    description="This endpoint starts the training process for the model.",
)
async def start_training():
    logging.info("Starting training process...")
    asyncio.create_task(train_service())
