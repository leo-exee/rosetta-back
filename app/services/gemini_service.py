import json
import re

from google.genai import types

from app.config.gemini_config import GeminiClient


def generate_with_gemini_service(prompt: str) -> str:
    response = GeminiClient.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="Tu es un assistant pédagogique expert en FLE (français langue étrangère)."
        ),
        contents=prompt,
    )
    return response.text or ""


def parse_json_list(raw_text: str):
    blocks = re.findall(r"\{[\s\S]*?\}", raw_text)
    results = []

    for block in blocks:
        try:
            match = re.search(r'"metadata"\s*:\s*\{[\s\S]*?\}', block)
            metadata_str = match.group(0)
            metadata = json.loads("{" + metadata_str + "}")

            input_match = re.search(r'"input"\s*:\s*"([^"]*)"', block)
            output_match = re.search(
                r'"output"\s*:\s*"([\s\S]*?)"\s*,\s*"metadata"', block
            )

            if input_match and output_match:
                input_str = input_match.group(1)
                output_str = output_match.group(1)
                results.append(
                    {"input": input_str, "output": output_str, "metadata": metadata}
                )
        except Exception as e:
            print(f"Failed to parse block:\n{block}\nError: {e}\n")
            continue
    return results
