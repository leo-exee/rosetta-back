from typing import Literal

from pydantic import BaseModel

from app.models.exercise_enum import ContextEnum, ExerciseTypeEnum, LevelEnum


class ExerciseInDTO(BaseModel):
    context: ContextEnum
    level: LevelEnum
    count: int = 3


class BaseExerciseDTO(BaseModel):
    type: ExerciseTypeEnum


class FillInTheBlanksDTO(BaseExerciseDTO):
    type: Literal[ExerciseTypeEnum.FILL_IN_THE_BLANKS] = (
        ExerciseTypeEnum.FILL_IN_THE_BLANKS
    )


class DefinitionMatcherDTO(BaseExerciseDTO):
    type: Literal[ExerciseTypeEnum.DEFINITION_MATCHER] = (
        ExerciseTypeEnum.DEFINITION_MATCHER
    )


class ComprehensionDTO(BaseExerciseDTO):
    type: Literal[ExerciseTypeEnum.COMPREHENSION] = ExerciseTypeEnum.COMPREHENSION


class ExerciseListOutDTO(BaseModel):
    context: ContextEnum
    Level: LevelEnum
    exercises: list[
        FillInTheBlanksDTO | DefinitionMatcherDTO | ComprehensionDTO | BaseExerciseDTO
    ]
