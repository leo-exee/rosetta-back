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
      "answer": "Je vais à la plage chaque été. Je prends ma valise et mon billet avant de partir.",
      "blanksCorrection": [
        "'Vais' is the first person singular form of 'aller' when talking about regular actions or habits",
        "We use 'plage' because the context mentions summer vacation and activities by the sea",
        "'Prends' expresses the action of taking/bringing items, conjugated for 'je'",
        "'Billet' mean ticket, the specific document needed for transportation",
        "'Partir' is the infinitive form required after the preposition 'de' to express departure"
      ]
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
      "answers": [
        { "word": "valise", "definition": "Objet pour transporter ses affaires", "translation": "Suitcase - Object used to carry personal belongings when traveling" },
        { "word": "billet", "definition": "Document pour embarquer", "translation": "Ticket - Document required to board transportation" },
        { "word": "avion", "definition": "Moyen de transport aérien", "translation": "Airplane - Aircraft used for air transportation" }
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
          "options": ["À la mer", "À la montagne", "En ville"],
          "explanation": "The text clearly states that Marie is going 'à la montagne' (to the mountains) for her vacation."
        },
        {
          "question": "Quel moyen de transport utilise-t-elle ?",
          "answer": "Le train",
          "options": ["Le bus", "La voiture", "Le train"],
          "explanation": "The sentence specifies that she 'prend le train' (takes the train) as her means of transportation."
        }
      ]
    }
""",
    }

    example_block = type_examples[input_dto.type]

    constraints_map = {
        ExerciseTypeEnum.FILL_IN_THE_BLANKS: f"""- Chaque exercice doit contenir environ {settings['fillInTheBlanks_blanks']} mots à compléter (blanks). Les blancs doivent être des mots clés pertinents pour le contexte (par exemple, des verbes, noms ou adjectifs).
- Fournis une correction explicative en anglais pour chaque blank dans le champ "blanksCorrection" expliquant POURQUOI ce mot est utilisé dans ce contexte (règle grammaticale, logique contextuelle, etc.). Pour l'emplacement des blancs, utilise ___ pour indiquer où le mot doit être inséré.""",
        ExerciseTypeEnum.DEFINITION_MATCHER: f"""- Chaque exercice doit contenir {settings['definitionMatcher_count']} mots et leurs définitions associées.
- Inclus une traduction complète en anglais de chaque mot et définition dans le champ "translation" pour faciliter la compréhension.""",
        ExerciseTypeEnum.COMPREHENSION: f"""- Chaque exercice doit contenir {settings['comprehension_questions']} questions de compréhension avec 3 options (dont la bonne réponse).
- Ajoute une explication détaillée en anglais pour chaque réponse dans le champ "explanation" en justifiant pourquoi c'est la bonne réponse.""",
    }

    constraints = constraints_map[input_dto.type]

    exercises_count = (
        "un exercice" if input_dto.count == 1 else f"{input_dto.count} exercices"
    )

    return f"""
Tu es un générateur d'exercices de français pour une application éducative destinée à des apprenants anglophones.

Consignes :
- Génère exactement {exercises_count} de type "{input_dto.type.value}".
- Le contenu doit être adapté au contexte : "{input_dto.context.value}" et au niveau : "{input_dto.level.value}".
- Tous les champs sont obligatoires, y compris les champs de correction et d'explication.
- Le niveau influe sur la complexité du vocabulaire, la syntaxe et les consignes.
- IMPORTANT: Le champ "exercises" doit TOUJOURS être un tableau, même pour un seul exercice.
- Toutes les explications et corrections doivent être rédigées en anglais pour aider les apprenants anglophones.

Contraintes spécifiques :
{constraints}

Format attendu (ne pas recopier l'exemple ci-dessous, il est là pour illustrer la structure) :

EXEMPLE :
{example_block}

Ta réponse doit respecter exactement cette structure (sans recopier l'exemple ci-dessus) :
{{
  "context": "{input_dto.context.value}",
  "Level": "{input_dto.level.value}",
  "exercises": [
    ... // tes exercices ici
  ]
}}


⚠️ RÈGLES CRITIQUES :
- Ne génère **aucune autre information** que le JSON (pas d'explication, pas de balise de code).
- Le champ "exercises" doit être un tableau même pour 1 seul exercice.
- Respecte exactement la structure JSON demandée.
- Tous les champs de correction sont obligatoires et doivent être remplis en anglais avec du contenu pertinent.
"""
