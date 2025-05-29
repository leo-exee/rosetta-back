from google.genai import types

from app.config.gemini_config import GeminiClient


def generate_gemini_content_service(prompt: str) -> str | None:
    response = GeminiClient.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="Tu es un assistant pédagogique expert en FLE (français langue étrangère)."
        ),
        contents=prompt,
    )
    return response.text
