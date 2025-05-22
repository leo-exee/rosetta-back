import logging
import random
import re
from enum import Enum

import torch
from fastapi import status
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.models.error_response import ErrorResponse


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ExerciseType(str, Enum):
    FILL_BLANK = "fill_blank"
    DEFINITION = "definition"
    REVERSE_DEFINITION = "reverse_definition"
    COMPREHENSION = "comprehension"
    MULTIPLE_CHOICE = "multiple_choice"
    TRANSLATION = "translation"


class ExerciseRequest(BaseModel):
    context: str  # e.g., "cooking", "working", "IT", "travel"
    level: DifficultyLevel
    exercise_type: ExerciseType
    count: int = 5
    custom_text: str | None = None


class Exercise(BaseModel):
    id: str
    type: ExerciseType
    question: str
    answer: str
    options: list[str] | None = None  # For multiple choice
    context: str
    level: DifficultyLevel
    difficulty_score: float


class ExerciseGenerationService:
    def __init__(self, model_path: str = "data/models/exercise_model"):
        self.model_path = model_path
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self._load_model()

        # Context-specific vocabulary and patterns
        self.context_vocabulary = {
            "cooking": {
                "beginner": [
                    "cook",
                    "eat",
                    "food",
                    "recipe",
                    "kitchen",
                    "meal",
                    "ingredient",
                ],
                "intermediate": [
                    "sauté",
                    "marinate",
                    "garnish",
                    "seasoning",
                    "utensil",
                    "preparation",
                ],
                "advanced": [
                    "julienne",
                    "emulsify",
                    "caramelize",
                    "confit",
                    "sous vide",
                    "molecular gastronomy",
                ],
            },
            "working": {
                "beginner": [
                    "work",
                    "job",
                    "office",
                    "meeting",
                    "colleague",
                    "boss",
                    "task",
                ],
                "intermediate": [
                    "deadline",
                    "project",
                    "teamwork",
                    "productivity",
                    "schedule",
                    "presentation",
                ],
                "advanced": [
                    "synergy",
                    "stakeholder",
                    "optimization",
                    "strategic planning",
                    "workflow automation",
                ],
            },
            "IT": {
                "beginner": [
                    "computer",
                    "internet",
                    "software",
                    "hardware",
                    "email",
                    "password",
                    "file",
                ],
                "intermediate": [
                    "database",
                    "server",
                    "network",
                    "programming",
                    "debugging",
                    "API",
                ],
                "advanced": [
                    "microservices",
                    "containerization",
                    "machine learning",
                    "blockchain",
                    "DevOps",
                ],
            },
            "travel": {
                "beginner": [
                    "travel",
                    "hotel",
                    "airport",
                    "ticket",
                    "luggage",
                    "passport",
                    "vacation",
                ],
                "intermediate": [
                    "itinerary",
                    "boarding pass",
                    "customs",
                    "currency exchange",
                    "accommodation",
                ],
                "advanced": [
                    "jet lag",
                    "cultural immersion",
                    "sustainable tourism",
                    "expedition",
                    "visa requirements",
                ],
            },
        }

        self.context_sentences = {
            "cooking": {
                "beginner": [
                    "I like to cook simple meals at home.",
                    "The recipe calls for basic ingredients.",
                    "We eat dinner together every evening.",
                ],
                "intermediate": [
                    "The chef demonstrated proper knife techniques.",
                    "Marinating the meat enhances its flavor significantly.",
                    "Proper seasoning is essential for balanced dishes.",
                ],
                "advanced": [
                    "The molecular gastronomy technique transforms ordinary ingredients.",
                    "Sous vide cooking ensures precise temperature control.",
                    "Emulsification creates stable foam textures.",
                ],
            },
            "working": {
                "beginner": [
                    "I go to work every morning at nine.",
                    "Our team has a meeting every Monday.",
                    "My colleague helps me with difficult tasks.",
                ],
                "intermediate": [
                    "The project deadline requires careful coordination.",
                    "Effective teamwork increases overall productivity.",
                    "Regular presentations keep stakeholders informed.",
                ],
                "advanced": [
                    "Strategic planning involves comprehensive market analysis.",
                    "Workflow automation reduces operational inefficiencies.",
                    "Synergistic collaboration maximizes competitive advantages.",
                ],
            },
            "IT": {
                "beginner": [
                    "I use my computer for work and entertainment.",
                    "The internet connects people around the world.",
                    "Remember to use a strong password.",
                ],
                "intermediate": [
                    "The database stores all customer information.",
                    "Network security prevents unauthorized access.",
                    "Programming requires logical thinking skills.",
                ],
                "advanced": [
                    "Microservices architecture improves system scalability.",
                    "Machine learning algorithms analyze complex patterns.",
                    "DevOps practices streamline deployment processes.",
                ],
            },
            "travel": {
                "beginner": [
                    "I love to travel to new places.",
                    "The hotel room was comfortable and clean.",
                    "Don't forget your passport at home.",
                ],
                "intermediate": [
                    "The detailed itinerary includes all planned activities.",
                    "Currency exchange rates affect travel budgets.",
                    "Comfortable accommodation enhances travel experiences.",
                ],
                "advanced": [
                    "Sustainable tourism minimizes environmental impact.",
                    "Cultural immersion provides authentic experiences.",
                    "Expedition planning requires extensive preparation.",
                ],
            },
        }

    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logging.warning(
                f"Could not load custom model: {e}. Using default t5-small."
            )
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.model.to(self.device)
            self.model.eval()

    def generate_exercises(self, request: ExerciseRequest) -> list[Exercise]:
        """Generate exercises based on the request parameters."""
        exercises = []

        # Get context-appropriate content
        content = self._get_context_content(request.context, request.level)

        for i in range(request.count):
            if request.exercise_type == ExerciseType.FILL_BLANK:
                exercise = self._generate_fill_blank(content, request, i)
            elif request.exercise_type == ExerciseType.DEFINITION:
                exercise = self._generate_definition(content, request, i)
            elif request.exercise_type == ExerciseType.REVERSE_DEFINITION:
                exercise = self._generate_reverse_definition(content, request, i)
            elif request.exercise_type == ExerciseType.MULTIPLE_CHOICE:
                exercise = self._generate_multiple_choice(content, request, i)
            elif request.exercise_type == ExerciseType.COMPREHENSION:
                exercise = self._generate_comprehension(content, request, i)
            else:
                exercise = self._generate_fill_blank(content, request, i)

            if exercise:
                exercises.append(exercise)

        return exercises

    def _get_context_content(self, context: str, level: DifficultyLevel) -> dict:
        """Get context-specific vocabulary and sentences."""
        context_lower = context.lower()

        if context_lower not in self.context_vocabulary:
            context_lower = "working"  # Default fallback

        return {
            "vocabulary": self.context_vocabulary[context_lower].get(level.value, []),
            "sentences": self.context_sentences[context_lower].get(level.value, []),
        }

    def _generate_fill_blank(
        self, content: dict, request: ExerciseRequest, index: int
    ) -> Exercise:
        """Generate a fill-in-the-blank exercise."""
        sentences = content["sentences"]
        content["vocabulary"]

        if not sentences:
            raise ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                status="NO_SENTENCES",
                message="No sentences available for the selected context and level.",
            )

        sentence = random.choice(sentences)
        words = sentence.split()

        # Find suitable words to blank out
        suitable_words = [
            (i, word)
            for i, word in enumerate(words)
            if len(word) > 3
            and word.lower() not in ["the", "and", "but", "for", "with"]
        ]

        if not suitable_words:
            raise ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                status="NO_SUITABLE_WORDS",
                message="No suitable words found for fill-in-the-blank exercise.",
            )

        word_index, target_word = random.choice(suitable_words)
        target_word_clean = re.sub(r"[^\w]", "", target_word)

        # Create blanked sentence
        blanked_words = words.copy()
        blanked_words[word_index] = "____"
        question = " ".join(blanked_words)

        difficulty_score = self._calculate_difficulty(target_word_clean, request.level)

        return Exercise(
            id=f"{request.context}_{request.exercise_type}_{index}",
            type=request.exercise_type,
            question=f"Fill in the blank: {question}",
            answer=target_word_clean,
            context=request.context,
            level=request.level,
            difficulty_score=difficulty_score,
        )

    def _generate_definition(
        self, content: dict, request: ExerciseRequest, index: int
    ) -> Exercise:
        """Generate a definition exercise."""
        vocabulary = content["vocabulary"]

        if not vocabulary:
            raise ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                status="NO_VOCABULARY",
                message="No vocabulary available for the selected context and level.",
            )

        word = random.choice(vocabulary)

        # Use model to generate definition or use predefined ones
        definition = self._get_definition(word, request.context, request.level)

        difficulty_score = self._calculate_difficulty(word, request.level)

        return Exercise(
            id=f"{request.context}_{request.exercise_type}_{index}",
            type=request.exercise_type,
            question=f"Define the word: {word}",
            answer=definition,
            context=request.context,
            level=request.level,
            difficulty_score=difficulty_score,
        )

    def _generate_reverse_definition(
        self, content: dict, request: ExerciseRequest, index: int
    ) -> Exercise:
        """Generate a reverse definition exercise."""
        vocabulary = content["vocabulary"]

        if not vocabulary:
            raise ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                status="NO_VOCABULARY",
                message="No vocabulary available for the selected context and level.",
            )

        word = random.choice(vocabulary)
        definition = self._get_definition(word, request.context, request.level)

        difficulty_score = self._calculate_difficulty(word, request.level)

        return Exercise(
            id=f"{request.context}_{request.exercise_type}_{index}",
            type=request.exercise_type,
            question=f"What word matches this definition: {definition}",
            answer=word,
            context=request.context,
            level=request.level,
            difficulty_score=difficulty_score,
        )

    def _generate_multiple_choice(
        self, content: dict, request: ExerciseRequest, index: int
    ) -> Exercise:
        """Generate a multiple choice exercise."""
        vocabulary = content["vocabulary"]

        if len(vocabulary) < 4:
            raise ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                status="NOT_ENOUGH_VOCABULARY",
                message="Not enough vocabulary for multiple choice exercise.",
            )

        correct_word = random.choice(vocabulary)
        definition = self._get_definition(correct_word, request.context, request.level)

        # Generate wrong options
        wrong_options = random.sample([w for w in vocabulary if w != correct_word], 3)
        options = [correct_word] + wrong_options
        random.shuffle(options)

        difficulty_score = self._calculate_difficulty(correct_word, request.level)

        return Exercise(
            id=f"{request.context}_{request.exercise_type}_{index}",
            type=request.exercise_type,
            question=f"Which word means: {definition}",
            answer=correct_word,
            options=options,
            context=request.context,
            level=request.level,
            difficulty_score=difficulty_score,
        )

    def _generate_comprehension(
        self, content: dict, request: ExerciseRequest, index: int
    ) -> Exercise:
        """Generate a comprehension exercise."""
        sentences = content["sentences"]

        if not sentences:
            raise ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                status="NO_SENTENCES",
                message="No sentences available for the selected context and level.",
            )

        sentence = random.choice(sentences)

        # Extract key information for question
        words = sentence.split()
        important_words = [w for w in words if len(w) > 4 and w[0].isupper()]

        if important_words:
            answer = important_words[0]
            question = f"Based on this text: '{sentence}' - What is the main subject mentioned?"
        else:
            # Fallback question
            question = f"What is the main idea in: '{sentence}'"
            answer = "Understanding the context"

        difficulty_score = self._calculate_difficulty(sentence, request.level)

        return Exercise(
            id=f"{request.context}_{request.exercise_type}_{index}",
            type=request.exercise_type,
            question=question,
            answer=answer,
            context=request.context,
            level=request.level,
            difficulty_score=difficulty_score,
        )

    def _get_definition(self, word: str, context: str, level: DifficultyLevel) -> str:
        """Get definition for a word, considering context and level."""
        # Predefined definitions based on context
        definitions = {
            "cooking": {
                "cook": "to prepare food by heating",
                "sauté": "to cook quickly in a small amount of fat",
                "julienne": "to cut food into thin strips",
                "marinate": "to soak food in seasoned liquid",
            },
            "working": {
                "deadline": "a time limit for completing a task",
                "synergy": "combined effort producing greater results",
                "productivity": "the efficiency of work output",
            },
            "IT": {
                "database": "organized collection of data",
                "microservices": "architectural approach using small services",
                "API": "interface for software communication",
            },
            "travel": {
                "itinerary": "planned route or journey schedule",
                "expedition": "organized journey for exploration",
                "customs": "border control procedures",
            },
        }

        context_defs = definitions.get(context.lower(), {})
        return context_defs.get(word, f"a {context}-related term: {word}")

    def _calculate_difficulty(self, text: str, level: DifficultyLevel) -> float:
        """Calculate difficulty score based on text complexity and level."""
        base_score = {
            DifficultyLevel.BEGINNER: 0.3,
            DifficultyLevel.INTERMEDIATE: 0.6,
            DifficultyLevel.ADVANCED: 0.9,
        }[level]

        # Adjust based on text length and complexity
        length_factor = min(len(text) / 20, 1.0)
        complexity_factor = len([c for c in text if c.isupper()]) / max(len(text), 1)

        return min(base_score + length_factor * 0.2 + complexity_factor * 0.1, 1.0)


# Service instance
exercise_service = ExerciseGenerationService()
