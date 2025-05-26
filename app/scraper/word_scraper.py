# app/scraper/word_scraper.py

import json
from pathlib import Path
from time import sleep

import requests

CONTEXTS = ["it", "work", "travel", "cooking"]
LEVELS = ["beginner", "intermediate", "advanced"]

INPUT_DIR = Path("data/words")
OUTPUT_FILE = Path("datasets/definitions.json")


def load_words_for_context(context: str) -> list[dict]:
    """Charge les mots depuis un fichier avec gestion des niveaux."""
    file_path = INPUT_DIR / f"{context}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"No file found for context: {context}")

    words_with_levels = []
    current_level = None

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#level:"):
                current_level = line.replace("#level:", "").strip()
                continue
            if current_level:
                words_with_levels.append(
                    {"word": line, "level": current_level, "context": context}
                )

    return words_with_levels


def fetch_word_data(word: str) -> dict | None:
    """Récupère les données de définition depuis dictionaryapi.dev"""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[!] Skipped {word} (not found)")
            return None
        data = response.json()[0]
        meanings = data.get("meanings", [])
        definitions, examples, synonyms = [], [], set()
        for meaning in meanings:
            for d in meaning.get("definitions", []):
                definitions.append(d.get("definition"))
                if d.get("example"):
                    examples.append(d["example"])
                synonyms.update(d.get("synonyms", []))
        return {
            "definitions": definitions,
            "examples": examples,
            "synonyms": list(synonyms),
        }
    except Exception as e:
        print(f"[!] Error fetching '{word}': {e}")
        return None


def scrape_all_definitions() -> list[dict]:
    """Scrape les définitions pour tous les contextes et niveaux."""
    all_data = []
    for context in CONTEXTS:
        print(f"🔍 Scraping context: {context}")
        try:
            words = load_words_for_context(context)
        except FileNotFoundError as e:
            print(e)
            continue

        for entry in words:
            word = entry["word"]
            print(f" → Fetching: {word} ({entry['level']})")
            word_data = fetch_word_data(word)
            if word_data:
                all_data.append(
                    {
                        "word": word,
                        "context": context,
                        "level": entry["level"],
                        **word_data,
                    }
                )
            sleep(1)
    return all_data


def save_definitions(data: list[dict]):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    data = scrape_all_definitions()
    save_definitions(data)
