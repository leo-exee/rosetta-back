from app.models.exercise_dto import ExerciseInDTO
from app.models.exercise_enum import ExerciseTypeEnum, LevelEnum


def build_prompt(input_dto: ExerciseInDTO) -> str:
    level_settings = {
        LevelEnum.BEGINNER: {
            "fillInTheBlanks_blanks": 3,
            "definitionMatcher_count": 3,
            "comprehension_questions": 3,
        },
        LevelEnum.INTERMEDIATE: {
            "fillInTheBlanks_blanks": 5,
            "definitionMatcher_count": 5,
            "comprehension_questions": 5,
        },
        LevelEnum.ADVANCED: {
            "fillInTheBlanks_blanks": 9,
            "definitionMatcher_count": 9,
            "comprehension_questions": 9,
        },
    }

    settings = level_settings[input_dto.level]

    type_examples = {
        ExerciseTypeEnum.FILL_IN_THE_BLANKS: """
    {
      "type": "fillInTheBlanks",
      "text": "Je [...] à la [...] chaque été. Je [...] ma valise et mon [...] avant de [...].",
      "blanks": ["vais", "plage", "prends", "billet", "partir"],
      "answer": "Je vais à la plage chaque été. Je prends ma valise et mon billet avant de partir."
    }
""",
        ExerciseTypeEnum.DEFINITION_MATCHER: """
    {
      "type": "definitionMatcher",
      "words": ["valise", "billet", "avion"],
      "definitions": [
        "Objet pour transporter ses affaires",
        "Document pour embarquer",
        "Moyen de transport aérien"
      ],
      "awswers": [
        { "word": "valise", "definition": "Objet pour transporter ses affaires" },
        { "word": "billet", "definition": "Document pour embarquer" },
        { "word": "avion", "definition": "Moyen de transport aérien" }
      ]
    }
""",
        ExerciseTypeEnum.COMPREHENSION: """
    {
      "type": "comprehension",
      "text": "Marie part en vacances à la montagne. Elle prend le train.",
      "questions": [
        {
          "question": "Où part Marie ?",
          "answer": "À la montagne",
          "options": ["À la mer", "À la montagne", "En ville"]
        },
        {
          "question": "Quel moyen de transport utilise-t-elle ?",
          "answer": "Le train",
          "options": ["Le bus", "La voiture", "Le train"]
        }
      ]
    }
""",
    }

    example_block = type_examples[input_dto.type]

    constraints_map = {
        ExerciseTypeEnum.FILL_IN_THE_BLANKS: f"- Chaque exercice doit contenir environ {settings['fillInTheBlanks_blanks']} mots à compléter (blanks).",
        ExerciseTypeEnum.DEFINITION_MATCHER: f"- Chaque exercice doit contenir {settings['definitionMatcher_count']} mots et leurs définitions associées.",
        ExerciseTypeEnum.COMPREHENSION: f"- Chaque exercice doit contenir {settings['comprehension_questions']} questions de compréhension avec 3 options (dont la bonne réponse).",
    }

    constraints = constraints_map[input_dto.type]

    return f"""
Tu es un générateur d'exercices de français pour une application éducative.

Consignes :
- Génère exactement {input_dto.count} exercices de type "{input_dto.type.value}".
- Le contenu doit être adapté au contexte : "{input_dto.context.value}" et au niveau : "{input_dto.level.value}".
- Tous les champs sont obligatoires.
- Le niveau influe sur la complexité du vocabulaire, la syntaxe et les consignes.

Contraintes spécifiques :
{constraints}

Format de sortie : JSON strictement conforme à ce modèle :

{{
  "context": "{input_dto.context.value}",
  "Level": "{input_dto.level.value}",
  "exercises": [
    {example_block}
  ]
}}

⚠️ Ne génère **aucune autre information** que le JSON (pas d’explication, pas de balise de code).
"""
