from fastapi import APIRouter

from app.models.exercise_dto import ExerciseInDTO
from app.services.exercice_service import (
    generate_definition_matchers,
    generate_fill_in_blank,
    generate_sentence_scrambler,
)

exercise_router = APIRouter(
    prefix="/exercises",
    tags=["batch-exercises"],
)


@exercise_router.post("/definition-matcher")
async def definition_matcher(req: ExerciseInDTO):
    return {
        "type": "definition_matcher",
        "exercises": await generate_definition_matchers(req),
    }


@exercise_router.post("/fill-in-blank")
async def fill_in_blank(req: ExerciseInDTO):
    return {"type": "fill_in_blank", "exercises": await generate_fill_in_blank(req)}


@exercise_router.post("/sentence-scrambler")
async def sentence_scrambler(req: ExerciseInDTO):
    return {
        "type": "sentence_scrambler",
        "exercises": await generate_sentence_scrambler(req),
    }
