import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GeminiClient = genai.Client(api_key=GEMINI_API_KEY)
