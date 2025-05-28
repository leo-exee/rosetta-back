import json
import logging
import os
import random
import sqlite3
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrenchRemedeFetcher:
    def __init__(self, db_path: str = "data/remede.db", delay_range: tuple = (0.5, 2.0)):
        self.delay_range = delay_range
        self.db_path = db_path
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_definitions": 0,
        }
        self.cache = {}

    def fetch_definition_remede(self, word: str) -> dict | None:
        try:
            self.stats["total_requests"] += 1
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT document FROM dictionary WHERE word = ? COLLATE NOCASE", (word,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                self.stats["failed_requests"] += 1
                return None

            doc = json.loads(row["document"])
            definitions = []
            examples = []

            for entry in doc.get("definitions", []):
                explanations = entry.get("explanations", [])
                for explanation in explanations:
                    definitions.append(
                        {
                            "definition": explanation,
                            "part_of_speech": entry.get("nature", ""),
                            "example": (
                                entry.get("examples", [""])[0]
                                if entry.get("examples")
                                else ""
                            ),
                        }
                    )
                    examples.extend(entry.get("examples", []))

            self.stats["successful_requests"] += 1
            return {
                "word": word,
                "definitions": definitions[:5],
                "examples": examples[:3],
                "source": "remede_sqlite",
                "language": "french",
            }

        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"❌ Erreur SQLite Remède pour '{word}': {e}")
            return None

    def get_french_definition(self, word: str) -> dict | None:
        clean_word = word.lower().strip()
        if clean_word in self.cache:
            self.stats["cached_definitions"] += 1
            return self.cache[clean_word]

        time.sleep(random.uniform(*self.delay_range))
        definition = self.fetch_definition_remede(clean_word)

        if definition and definition.get("definitions"):
            self.cache[clean_word] = definition
            return definition
        else:
            return None

    def process_french_keywords_file(
        self, keywords_file: str, context: str
    ) -> list[dict]:
        logger.info(f"🇫🇷 Traitement du fichier de mots-clés français: {keywords_file}")

        if not os.path.exists(keywords_file):
            logger.error(f"❌ Fichier non trouvé: {keywords_file}")
            return []

        with open(keywords_file, encoding="utf-8") as f:
            keywords = json.load(f)

        logger.info(
            f"📝 {len(keywords)} mots-clés français à enrichir pour '{context}'"
        )

        enriched_keywords = []

        for i, keyword in enumerate(keywords):
            word = keyword["word"]

            try:
                definition_data = self.get_french_definition(word)

                enriched_entry = {
                    "word": word,
                    "context": context,
                    "level": keyword["level"],
                    "global_frequency": keyword.get("global_frequency", 0),
                    "importance_score": keyword.get("importance_score", 0),
                    "pos_tags": keyword.get("pos_tags", []),
                    "contexts_from_articles": keyword.get("contexts", []),
                    "source_articles": [
                        {
                            "url": keyword.get("source_url", ""),
                            "title": keyword.get("source_title", ""),
                        }
                    ],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "language": "french",
                }

                if definition_data:
                    enriched_entry.update(
                        {
                            "definitions": definition_data.get("definitions", []),
                            "examples_from_dict": definition_data.get("examples", []),
                            "dictionary_source": definition_data.get("source", ""),
                            "has_definition": True,
                            "definition_count": len(
                                definition_data.get("definitions", [])
                            ),
                        }
                    )
                else:
                    enriched_entry.update(
                        {
                            "definitions": [],
                            "examples_from_dict": [],
                            "dictionary_source": "",
                            "has_definition": False,
                            "definition_count": 0,
                        }
                    )

                enriched_keywords.append(enriched_entry)

                if (i + 1) % 10 == 0:
                    success_rate = (
                        self.stats["successful_requests"]
                        / max(self.stats["total_requests"], 1)
                    ) * 100
                    logger.info(
                        f"✅ {i + 1}/{len(keywords)} mots français traités (succès: {success_rate:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"❌ Erreur pour le mot français '{word}': {e}")
                continue

        total_with_def = sum(1 for kw in enriched_keywords if kw["has_definition"])
        success_rate = (
            (total_with_def / len(enriched_keywords)) * 100 if enriched_keywords else 0
        )

        logger.info(
            f"🎯 {len(enriched_keywords)} mots français traités pour '{context}'"
        )
        logger.info(
            f"📊 {total_with_def} mots avec définition Remède ({success_rate:.1f}%)"
        )

        return enriched_keywords

    def save_french_definitions(self, definitions: list[dict], output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(definitions, f, ensure_ascii=False, indent=2)
        logger.info(
            f"💾 {len(definitions)} définitions françaises sauvegardées dans {output_file}"
        )

    def print_french_stats(self):
        logger.info("📊 Statistiques Remède SQLite:")
        logger.info(f"   Total requêtes: {self.stats['total_requests']}")
        logger.info(f"   Succès: {self.stats['successful_requests']}")
        logger.info(f"   Échecs: {self.stats['failed_requests']}")
        logger.info(f"   Cache utilisé: {self.stats['cached_definitions']}")


def main():
    fetcher = FrenchRemedeFetcher(db_path="data/remede.db", delay_range=(1.0, 2.0))

    keywords_dir = "datasets/keywords_fr"
    definitions_dir = "datasets/definitions_fr"
    os.makedirs(definitions_dir, exist_ok=True)

    contexts = ["it", "work", "travel", "cooking"]

    for context in contexts:
        keywords_file = f"{keywords_dir}/{context}_keywords_fr.json"

        if os.path.exists(keywords_file):
            logger.info(f"🚀 Récupération des définitions françaises pour '{context}'")

            definitions = fetcher.process_french_keywords_file(keywords_file, context)

            if definitions:
                output_file = f"{definitions_dir}/{context}_definitions_fr.json"
                fetcher.save_french_definitions(definitions, output_file)
            else:
                logger.warning(
                    f"⚠️ Aucune définition française récupérée pour '{context}'"
                )
        else:
            logger.warning(
                f"⚠️ Fichier de mots-clés français non trouvé: {keywords_file}"
            )

    fetcher.print_french_stats()
    logger.info("🎉 Récupération française terminée pour tous les contextes")


if __name__ == "__main__":
    main()
