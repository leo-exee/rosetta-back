from app.models.exercise_dto import ExerciseInDTO
from app.services.gemini_service import generate_with_gemini_service, parse_json_list


async def generate_definition_matchers(req: ExerciseInDTO):
    prompt = f"""Génère {req.count} exercices de type "Definition Matcher" pour des apprenants {req.level} en français dans le contexte "{req.context}".
Format JSON :
{{"input":"context|level","output":"mots|||définitions|||ordre","metadata":{{"context":"context","level":"level","words":[],"original_definitions":[],"language":"french"}}}}"""
    return parse_json_list(generate_with_gemini_service(prompt))


async def generate_fill_in_blank(req: ExerciseInDTO):
    prompt = f"""Génère {req.count} exercices de type "Fill in the blank" pour des apprenants {req.level} en français dans le contexte "{req.context}".
Format JSON :
{{"input":"context|level|phrase","output":"phrase_incomplete|||distracteurs|||phrase_complete","metadata":{{"context":"context","level":"level","source":"article","difficulty_words":[],"language":"french"}}}}"""
    return parse_json_list(generate_with_gemini_service(prompt))


async def generate_sentence_scrambler(req: ExerciseInDTO):
    prompt = f"""Génère {req.count} exercices de type "Sentence Scrambler" pour des apprenants {req.level} en français dans le contexte "{req.context}".
Format JSON :
{{"input":"context|level|phrase","output":"mélangée|||originale","metadata":{{"context":"context","level":"level","word_count":0,"complexity":0.0,"language":"french"}}}}"""
    return parse_json_list(generate_with_gemini_service(prompt))
