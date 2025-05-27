import json
import logging
import os
import random
import re
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrenchWiktionaryFetcher:
    def __init__(self, delay_range: tuple = (0.5, 2.0)):
        """
        Récupérateur de définitions depuis Wiktionnaire français
        """
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; FrenchLearningBot/1.0; Educational purposes)",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            }
        )

        # URLs de base Wiktionnaire français
        self.wiktionary_api = "https://fr.wiktionary.org/api/rest_v1/page/definition/"
        self.wiktionary_web = "https://fr.wiktionary.org/wiki/"

        # Statistiques
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_definitions": 0,
            "api_requests": 0,
            "scrape_requests": 0,
        }

        # Cache pour éviter les requêtes répétées
        self.cache = {}

    def fetch_definition_wiktionary_api(self, word: str) -> dict | None:
        """
        Récupère via l'API REST de Wiktionnaire français
        """
        url = f"{self.wiktionary_api}{quote(word)}"

        try:
            self.stats["api_requests"] += 1
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self.parse_wiktionary_api_response(data, word)
            elif response.status_code == 404:
                logger.debug(f"🔍 Mot '{word}' non trouvé dans l'API Wiktionnaire")
                return None
            else:
                logger.warning(
                    f"⚠️ Erreur API Wiktionnaire pour '{word}': {response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"❌ Erreur API Wiktionnaire pour '{word}': {e}")
            return None

    def fetch_definition_wiktionary_scrape(self, word: str) -> dict | None:
        """
        Récupère via scraping de la page Wiktionnaire (fallback)
        """
        url = f"{self.wiktionary_web}{quote(word)}"

        try:
            self.stats["scrape_requests"] += 1
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                return self.parse_wiktionary_page(soup, word)
            else:
                return None

        except Exception as e:
            logger.error(f"❌ Erreur scraping Wiktionnaire pour '{word}': {e}")
            return None

    def parse_wiktionary_api_response(self, data: dict, word: str) -> dict:
        """Parse la réponse de l'API Wiktionnaire français"""
        definitions = []
        examples = []

        # Structure de l'API Wiktionnaire française
        if "fr" in data:
            french_data = data["fr"]

            for definition_entry in french_data:
                part_of_speech = definition_entry.get("partOfSpeech", "")

                if "definitions" in definition_entry:
                    for def_item in definition_entry["definitions"]:
                        definition_text = def_item.get("definition", "")
                        if definition_text:
                            # Nettoyer la définition
                            clean_def = self.clean_french_definition_text(
                                definition_text
                            )

                            definitions.append(
                                {
                                    "definition": clean_def,
                                    "part_of_speech": part_of_speech,
                                    "example": (
                                        def_item.get("examples", [""])[0]
                                        if def_item.get("examples")
                                        else ""
                                    ),
                                }
                            )

                            # Extraire les exemples
                            if def_item.get("examples"):
                                examples.extend(def_item["examples"][:2])

        return {
            "word": word,
            "definitions": definitions[:5],  # Max 5 définitions
            "examples": examples[:3],  # Max 3 exemples
            "source": "wiktionary_api",
            "language": "french",
        }

    def parse_wiktionary_page(self, soup: BeautifulSoup, word: str) -> dict:
        """Parse une page Wiktionnaire française scrapée"""
        definitions = []
        examples = []

        # Chercher la section française
        french_section = None
        for h2 in soup.find_all("h2"):
            span = h2.find("span", {"id": "Français"})
            if span:
                french_section = h2.parent
                break

        if not french_section:
            # Alternative: chercher directement les sections de définition
            french_section = soup

        # Chercher les définitions dans les listes ordonnées
        definition_count = 0
        for ol in french_section.find_all("ol"):
            if definition_count >= 5:  # Limiter à 5 définitions
                break

            for li in ol.find_all("li"):
                if definition_count >= 5:
                    break

                def_text = li.get_text().strip()
                if def_text and len(def_text) > 10 and not def_text.startswith("("):
                    clean_def = self.clean_french_definition_text(def_text)
                    if len(clean_def) > 5:  # Définition valide
                        definitions.append(
                            {
                                "definition": clean_def,
                                "part_of_speech": self.extract_part_of_speech(li),
                                "example": "",
                            }
                        )
                        definition_count += 1

        # Chercher des exemples dans les sections d'exemples
        for em in french_section.find_all("em", limit=3):
            example_text = em.get_text().strip()
            if (
                example_text
                and len(example_text) > 10
                and word.lower() in example_text.lower()
            ):
                examples.append(example_text)

        # Chercher aussi dans les sections avec class="example"
        for example_elem in french_section.find_all(class_="example"):
            example_text = example_elem.get_text().strip()
            if example_text and len(example_text) > 10:
                examples.append(example_text)

        return {
            "word": word,
            "definitions": definitions,
            "examples": examples[:3],
            "source": "wiktionary_scrape",
            "language": "french",
        }

    def extract_part_of_speech(self, li_element) -> str:
        """Extrait la nature grammaticale depuis un élément li"""
        # Chercher les abréviations grammaticales françaises courantes
        text = li_element.get_text().lower()

        pos_patterns = {
            "nom": r"\b(nom|n\.)\b",
            "verbe": r"\b(verbe|v\.)\b",
            "adjectif": r"\b(adjectif|adj\.)\b",
            "adverbe": r"\b(adverbe|adv\.)\b",
            "préposition": r"\b(préposition|prép\.)\b",
            "conjonction": r"\b(conjonction|conj\.)\b",
            "interjection": r"\b(interjection|interj\.)\b",
        }

        for pos, pattern in pos_patterns.items():
            if re.search(pattern, text):
                return pos

        return ""

    def clean_french_definition_text(self, text: str) -> str:
        """Nettoie le texte d'une définition française"""
        # Supprimer les références et liens
        text = re.sub(r"\[\d+\]", "", text)  # Références [1], [2], etc.
        text = re.sub(r"\([^)]*\)", "", text)  # Texte entre parenthèses
        text = re.sub(r"→ voir.*", "", text)  # Liens "voir aussi"
        text = re.sub(r"Voir aussi.*", "", text)  # Liens "voir aussi"

        # Supprimer les balises wiki
        text = re.sub(r"\{\{[^}]*\}\}", "", text)
        text = re.sub(r"\[\[[^\]]*\]\]", "", text)

        # Nettoyer les espaces
        text = re.sub(r"\s+", " ", text).strip()

        # Supprimer les préfixes courants français
        prefixes = ["Définition:", "Sens:", "•", "-", "*", "1.", "2.", "3."]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        # Supprimer les suffixes de référence
        text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

        return text.strip()

    def get_french_definition(self, word: str) -> dict | None:
        """
        Récupère la définition d'un mot français depuis Wiktionnaire
        """
        clean_word = word.lower().strip()

        # Vérifier le cache
        if clean_word in self.cache:
            self.stats["cached_definitions"] += 1
            return self.cache[clean_word]

        # Délai anti-spam
        time.sleep(random.uniform(*self.delay_range))
        self.stats["total_requests"] += 1

        # Tenter l'API d'abord (plus fiable)
        definition = self.fetch_definition_wiktionary_api(clean_word)

        # Fallback sur le scraping si l'API échoue ou donne peu de résultats
        if (
            not definition
            or not definition.get("definitions")
            or len(definition["definitions"]) == 0
        ):
            logger.debug(f"🔄 Fallback scraping pour '{word}'")
            definition = self.fetch_definition_wiktionary_scrape(clean_word)

        if (
            definition
            and definition.get("definitions")
            and len(definition["definitions"]) > 0
        ):
            self.stats["successful_requests"] += 1
            self.cache[clean_word] = definition
            logger.debug(
                f"✅ Définition française trouvée pour '{word}' ({len(definition['definitions'])} définitions)"
            )
            return definition
        else:
            self.stats["failed_requests"] += 1
            logger.debug(f"❌ Aucune définition française pour '{word}'")
            return None

    def process_french_keywords_file(
        self, keywords_file: str, context: str
    ) -> list[dict]:
        """
        Traite un fichier de mots-clés français et enrichit avec Wiktionnaire
        """
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
                # Récupérer depuis Wiktionnaire français
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

                # Log de progression
                if (i + 1) % 10 == 0:
                    success_rate = (
                        self.stats["successful_requests"]
                        / max(self.stats["total_requests"], 1)
                    ) * 100
                    logger.info(
                        f"✅ {i + 1}/{len(keywords)} mots français traités "
                        f"(succès: {success_rate:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"❌ Erreur pour le mot français '{word}': {e}")
                continue

        # Statistiques finales
        total_with_def = sum(1 for kw in enriched_keywords if kw["has_definition"])
        success_rate = (
            (total_with_def / len(enriched_keywords)) * 100 if enriched_keywords else 0
        )

        logger.info(
            f"🎯 {len(enriched_keywords)} mots français traités pour '{context}'"
        )
        logger.info(
            f"📊 {total_with_def} mots avec définition Wiktionnaire ({success_rate:.1f}%)"
        )

        return enriched_keywords

    def save_french_definitions(self, definitions: list[dict], output_file: str):
        """Sauvegarde les définitions françaises enrichies"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(definitions, f, ensure_ascii=False, indent=2)

        logger.info(
            f"💾 {len(definitions)} définitions françaises sauvegardées dans {output_file}"
        )

    def print_french_stats(self):
        """Affiche les statistiques détaillées Wiktionnaire français"""
        logger.info("📊 Statistiques Wiktionnaire français:")
        logger.info(f"   Total requêtes: {self.stats['total_requests']}")
        logger.info(f"   Succès: {self.stats['successful_requests']}")
        logger.info(f"   Échecs: {self.stats['failed_requests']}")
        logger.info(f"   Cache utilisé: {self.stats['cached_definitions']}")
        logger.info(f"   Requêtes API: {self.stats['api_requests']}")
        logger.info(f"   Requêtes scraping: {self.stats['scrape_requests']}")

        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_requests"] / self.stats["total_requests"]
            ) * 100
            logger.info(f"   Taux de succès global: {success_rate:.1f}%")


def main():
    """Test du service Wiktionnaire français"""
    fetcher = FrenchWiktionaryFetcher(delay_range=(1.0, 2.0))

    # Répertoires français
    keywords_dir = "datasets/keywords_fr"
    definitions_dir = "datasets/definitions_fr"
    os.makedirs(definitions_dir, exist_ok=True)

    # Traiter chaque contexte français
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

    # Afficher les statistiques
    fetcher.print_french_stats()
    logger.info("🎉 Récupération française terminée pour tous les contextes")


if __name__ == "__main__":
    main()
