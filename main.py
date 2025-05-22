import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.app_v1 import v1

sentry_env = os.getenv("SENTRY_ENVIRONMENT", "development")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", v1)
