import time

from google.genai import types
from google.genai.errors import ServerError

from app.config.gemini_config import GeminiClient


def generate_gemini_content_service(
    prompt: str, max_retries: int = 3, delay: float = 1.0
) -> str | None:

    for attempt in range(max_retries):
        try:
            response = GeminiClient.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                config=types.GenerateContentConfig(
                    system_instruction="Tu es un assistant pédagogique expert en FLE (français langue étrangère)."
                ),
                contents=prompt,
            )
            return response.text

        except ServerError as e:
            if hasattr(e, "code") and e.code == 503 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 1.5
                continue
            else:
                raise e
        except Exception as e:
            raise e
    return None
