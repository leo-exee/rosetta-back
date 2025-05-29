from enum import Enum


class ContextEnum(str, Enum):
    TRAVEL = "travel"
    CULTURE = "culture"
    KITCHEN = "kitchen"
    LITERATURE = "literature"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    GENERAL = "general"


class LevelEnum(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ExerciseTypeEnum(str, Enum):
    FILL_IN_THE_BLANKS = "fillInTheBlanks"
    DEFINITION_MATCHER = "definitionMatcher"
    COMPREHENSION = "comprehension"
