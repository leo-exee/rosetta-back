from fastapi import APIRouter

scraper_controller = APIRouter(
    prefix="/scraper",
    tags=["scraper"],
)
