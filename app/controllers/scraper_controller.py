import asyncio

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.services.scraper_service import scrape_data_service

scraper_controller = APIRouter(
    prefix="/scraper",
    tags=["scraper"],
)


@scraper_controller.post(
    "/start",
    summary="Start scraping",
    description="This endpoint starts the scraping process.",
)
async def start_scraping():
    print("Starting scraping process...")
    asyncio.create_task(scrape_data_service())
    return JSONResponse(
        content={"message": "Training process started."},
        status_code=status.HTTP_202_ACCEPTED,
    )
