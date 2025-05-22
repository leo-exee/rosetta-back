from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.controllers.scraper_controller import scraper_controller
from app.controllers.training_controller import training_controller
from app.models.error_response import ErrorResponse

v1 = FastAPI(
    title="Rosetta API",
    description=("Rosetta API for generating english exercises"),
    version="1.0.0",
    redoc_url="/redoc",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

v1.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@v1.exception_handler(Exception)
def error_response_handler(request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, ErrorResponse):
        return JSONResponse(
            status_code=int(error.code), content=jsonable_encoder(error)
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(
            ErrorResponse(
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status="error",
                message="Internal Server Error",
                errors=[str(error)],
            )
        ),
    )


v1.include_router(training_controller)
v1.include_router(scraper_controller)
