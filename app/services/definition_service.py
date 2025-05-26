import json
import logging
import os
import random
import time

import requests

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefinitionFetcher:
    def __init__(self, delay_range: tuple = (0.5, 2.0)):
        """
        Initialise le récupérateur de définitions
        Args:
            delay_range: Délai aléatoire entre les requêtes (min, max) en secondes
        """
        self.delay_range = delay_range
        self.session = requests.Session()

        # APIs de dictionnaires disponibles
        self.apis = {
            "dictionaryapi": "https://api.dictionaryapi.dev/api/v2/entries/en/",
            "wordnik": "https://api.wordnik.com/v4/word.json/",  # Nécessite une clé API
        }

        # Statistiques
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_definitions": 0,
        }

        # Cache simple pour éviter les requêtes répétées
        self.cache = {}

    def fetch_definition_dictionaryapi(self, word: str) -> dict | None:
        """
        Récupère la définition via dictionaryapi.dev (gratuite, sans clé)
        Args:
            word: Mot à chercher
        Returns:
            Dictionnaire avec définition et exemples ou None
        """
        url = f"{self.apis['dictionaryapi']}{word.lower()}"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return self.parse_dictionaryapi_response(data[0])

            elif response.status_code == 404:
                logger.debug(f"🔍 Mot '{word}' non trouvé dans l'API")
                return None

            else:
                logger.warning(f"⚠️ Erreur API pour '{word}': {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur réseau pour '{word}': {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur JSON pour '{word}': {e}")
            return None

    def parse_dictionaryapi_response(self, data: dict) -> dict:
        """
        Parse la réponse de dictionaryapi.dev
        Args:
            data: Réponse JSON de l'API
        Returns:
            Dictionnaire normalisé
        """
        word = data.get("word", "")
        phonetic = data.get("phonetic", "")

        definitions = []
        examples = []
        synonyms = []
        antonyms = []

        # Parcourir les significations
        for meaning in data.get("meanings", []):
            part_of_speech = meaning.get("partOfSpeech", "")

            # Récupérer les définitions
            for definition in meaning.get("definitions", []):
                def_text = definition.get("definition", "")
                if def_text:
                    definitions.append(
                        {
                            "definition": def_text,
                            "part_of_speech": part_of_speech,
                            "example": definition.get("example", ""),
                        }
                    )

                # Exemples
                if definition.get("example"):
                    examples.append(definition["example"])

            # Synonymes et antonymes
            synonyms.extend(meaning.get("synonyms", []))
            antonyms.extend(meaning.get("antonyms", []))

        return {
            "word": word,
            "phonetic": phonetic,
            "definitions": definitions,
            "examples": list(set(examples))[:3],  # Max 3 exemples uniques
            "synonyms": list(set(synonyms))[:5],  # Max 5 synonymes
            "antonyms": list(set(antonyms))[:5],  # Max 5 antonymes
            "source": "dictionaryapi.dev",
        }

    def get_definition(self, word: str) -> dict | None:
        """
        Récupère la définition d'un mot (avec cache)
        Args:
            word: Mot à chercher
        Returns:
            Dictionnaire avec définition ou None
        """
        # Normaliser le mot
        clean_word = word.lower().strip()

        # Vérifier le cache
        if clean_word in self.cache:
            self.stats["cached_definitions"] += 1
            return self.cache[clean_word]

        # Délai pour éviter le rate limiting
        time.sleep(random.uniform(*self.delay_range))

        self.stats["total_requests"] += 1

        # Tenter de récupérer la définition
        definition = self.fetch_definition_dictionaryapi(clean_word)

        if definition:
            self.stats["successful_requests"] += 1
            self.cache[clean_word] = definition
            logger.debug(f"✅ Définition trouvée pour '{word}'")
            return definition
        else:
            self.stats["failed_requests"] += 1
            logger.debug(f"❌ Aucune définition pour '{word}'")
            return None

    def process_keywords_file(self, keywords_file: str, context: str) -> list[dict]:
        """
        Traite un fichier de mots-clés et enrichit avec des définitions
        Args:
            keywords_file: Chemin vers le fichier JSON des mots-clés
            context: Contexte (it, work, travel)
        Returns:
            Liste des mots enrichis avec définitions
        """
        logger.info(f"🔍 Traitement du fichier: {keywords_file}")

        if not os.path.exists(keywords_file):
            logger.error(f"❌ Fichier non trouvé: {keywords_file}")
            return []

        # Charger les mots-clés
        with open(keywords_file, encoding="utf-8") as f:
            keywords = json.load(f)

        logger.info(f"📝 {len(keywords)} mots-clés à enrichir pour '{context}'")

        enriched_keywords = []

        for i, keyword in enumerate(keywords):
            word = keyword["word"]

            try:
                # Récupérer la définition
                definition_data = self.get_definition(word)

                # Créer l'entrée enrichie
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
                }

                # Ajouter les données de définition si trouvées
                if definition_data:
                    enriched_entry.update(
                        {
                            "phonetic": definition_data.get("phonetic", ""),
                            "definitions": definition_data.get("definitions", []),
                            "examples_from_dict": definition_data.get("examples", []),
                            "synonyms": definition_data.get("synonyms", []),
                            "antonyms": definition_data.get("antonyms", []),
                            "dictionary_source": definition_data.get("source", ""),
                            "has_definition": True,
                        }
                    )
                else:
                    enriched_entry.update(
                        {
                            "phonetic": "",
                            "definitions": [],
                            "examples_from_dict": [],
                            "synonyms": [],
                            "antonyms": [],
                            "dictionary_source": "",
                            "has_definition": False,
                        }
                    )

                enriched_keywords.append(enriched_entry)

                # Log de progression
                if (i + 1) % 20 == 0:
                    success_rate = (
                        self.stats["successful_requests"]
                        / max(self.stats["total_requests"], 1)
                    ) * 100
                    logger.info(
                        f"✅ {i + 1}/{len(keywords)} mots traités "
                        f"(succès: {success_rate:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"❌ Erreur pour le mot '{word}': {e}")
                continue

        # Statistiques finales
        total_with_def = sum(1 for kw in enriched_keywords if kw["has_definition"])
        success_rate = (
            (total_with_def / len(enriched_keywords)) * 100 if enriched_keywords else 0
        )

        logger.info(f"🎯 {len(enriched_keywords)} mots traités pour '{context}'")
        logger.info(f"📊 {total_with_def} mots avec définition ({success_rate:.1f}%)")

        return enriched_keywords

    def save_definitions(self, definitions: list[dict], output_file: str):
        """Sauvegarde les définitions enrichies"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(definitions, f, ensure_ascii=False, indent=2)

        logger.info(
            f"💾 {len(definitions)} définitions sauvegardées dans {output_file}"
        )

    def print_stats(self):
        """Affiche les statistiques d'utilisation"""
        logger.info("📊 Statistiques des requêtes:")
        logger.info(f"   Total: {self.stats['total_requests']}")
        logger.info(f"   Succès: {self.stats['successful_requests']}")
        logger.info(f"   Échecs: {self.stats['failed_requests']}")
        logger.info(f"   Cache: {self.stats['cached_definitions']}")

        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_requests"] / self.stats["total_requests"]
            ) * 100
            logger.info(f"   Taux de succès: {success_rate:.1f}%")


def main():
    """Fonction principale pour tester le récupérateur"""
    fetcher = DefinitionFetcher(delay_range=(1.0, 2.0))

    # Répertoires
    keywords_dir = "datasets/keywords"
    definitions_dir = "datasets/definitions"
    os.makedirs(definitions_dir, exist_ok=True)

    # Traiter chaque contexte
    contexts = ["it", "work", "travel"]

    for context in contexts:
        keywords_file = f"{keywords_dir}/{context}_keywords.json"

        if os.path.exists(keywords_file):
            logger.info(f"🚀 Récupération des définitions pour '{context}'")

            definitions = fetcher.process_keywords_file(keywords_file, context)

            if definitions:
                output_file = f"{definitions_dir}/{context}_definitions.json"
                fetcher.save_definitions(definitions, output_file)
            else:
                logger.warning(f"⚠️ Aucune définition récupérée pour '{context}'")
        else:
            logger.warning(f"⚠️ Fichier de mots-clés non trouvé: {keywords_file}")

    # Afficher les statistiques
    fetcher.print_stats()
    logger.info("🎉 Récupération terminée pour tous les contextes")


if __name__ == "__main__":
    main()
