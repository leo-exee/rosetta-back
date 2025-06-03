from typing import Literal

from pydantic import BaseModel, Field

from app.models.exercise_enum import ContextEnum, ExerciseTypeEnum, LevelEnum


class ExerciseInDTO(BaseModel):
    type: ExerciseTypeEnum
    context: ContextEnum
    level: LevelEnum
    count: int = 3


class BaseExerciseDTO(BaseModel):
    type: ExerciseTypeEnum


class FillInTheBlanksDTO(BaseExerciseDTO):
    type: Literal[ExerciseTypeEnum.FILL_IN_THE_BLANKS] = (
        ExerciseTypeEnum.FILL_IN_THE_BLANKS
    )
    text: str
    blanks: list[str]
    answer: str
    blanksCorrection: list[str] = Field(
        default=[],
        description="List of corrections for the blanks, used for the correction.",
    )


class DefinitionCorrectionDTO(BaseModel):
    word: str
    definition: str
    translation: str = Field(
        default="",
        description="Translation of the word and definition, used for the correction.",
    )


class DefinitionMatcherDTO(BaseExerciseDTO):
    type: Literal[ExerciseTypeEnum.DEFINITION_MATCHER] = (
        ExerciseTypeEnum.DEFINITION_MATCHER
    )
    words: list[str]
    definitions: list[str]
    answers: list[DefinitionCorrectionDTO]


class QuestionDTO(BaseModel):
    question: str
    answer: str
    explanation: str = Field(
        default="",
        description="Explanation of the answer, used for the correction.",
    )
    options: list[str] = Field(
        description="List of options for the question, including the correct answer."
    )


class ComprehensionDTO(BaseExerciseDTO):
    type: Literal[ExerciseTypeEnum.COMPREHENSION] = ExerciseTypeEnum.COMPREHENSION
    text: str
    questions: list[QuestionDTO]


class ExerciseListOutDTO(BaseModel):
    context: ContextEnum
    Level: LevelEnum
    exercises: list[
        FillInTheBlanksDTO | DefinitionMatcherDTO | ComprehensionDTO | BaseExerciseDTO
    ]
