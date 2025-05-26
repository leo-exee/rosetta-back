import json
import logging
import os
import random
import re
from collections import defaultdict

import spacy
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIDatasetGenerator:
    """
    Générateur de datasets pour entraîner les modèles d'IA
    qui créeront les exercices d'anglais
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.contexts = ["it", "work", "travel"]
        self.levels = ["beginner", "intermediate", "advanced"]

        # Chemins des données sources
        self.articles_dir = "data/articles"
        self.keywords_dir = "datasets/keywords"
        self.definitions_dir = "datasets/definitions"

        # Chemin de sortie pour les datasets d'entraînement
        self.output_dir = "datasets/training"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_source_data(self) -> dict:
        """Charge toutes les données sources"""
        data = {"articles": {}, "keywords": {}, "definitions": {}}

        for context in self.contexts:
            # Articles
            articles_file = f"{self.articles_dir}/{context}/articles.jsonl"
            if os.path.exists(articles_file):
                with open(articles_file, encoding="utf-8") as f:
                    data["articles"][context] = [
                        json.loads(line) for line in f if line.strip()
                    ]

            # Keywords
            keywords_file = f"{self.keywords_dir}/{context}_keywords.json"
            if os.path.exists(keywords_file):
                with open(keywords_file, encoding="utf-8") as f:
                    data["keywords"][context] = json.load(f)

            # Definitions
            definitions_file = f"{self.definitions_dir}/{context}_definitions.json"
            if os.path.exists(definitions_file):
                with open(definitions_file, encoding="utf-8") as f:
                    data["definitions"][context] = json.load(f)

        logger.info("✅ Données sources chargées")
        return data

    def generate_fill_in_blank_dataset(self, data: dict) -> list[dict]:
        """
        Génère le dataset pour le modèle Fill-in-the-Blank
        Format: {"input": "context|level|text", "output": "text_with_blanks|||words_to_fill|||complete_text"}
        """
        dataset = []

        for context in self.contexts:
            articles = data["articles"].get(context, [])
            keywords_by_level = defaultdict(list)

            # Organiser les mots-clés par niveau
            for kw in data["keywords"].get(context, []):
                keywords_by_level[kw["level"]].append(kw)

            for article in articles[:50]:  # Limite pour éviter trop de données
                content = article.get("content", "")

                # Extraire des phrases de longueur appropriée
                sentences = self.extract_suitable_sentences(content)

                for sentence in sentences[:5]:  # Max 5 phrases par article
                    for level in self.levels:
                        relevant_words = keywords_by_level[level]
                        if not relevant_words:
                            continue

                        # Créer des variantes avec différents mots supprimés
                        blanked_versions = self.create_fill_in_blank_variants(
                            sentence, relevant_words, level
                        )

                        for variant in blanked_versions:
                            dataset.append(
                                {
                                    "input": f"{context}|{level}|{sentence}",
                                    "output": f"{variant['text_with_blanks']}|||{variant['words_to_fill']}|||{sentence}",
                                    "metadata": {
                                        "context": context,
                                        "level": level,
                                        "source": "article",
                                        "difficulty_words": variant["target_words"],
                                    },
                                }
                            )

        logger.info(f"📝 {len(dataset)} exemples générés pour Fill-in-the-Blank")
        return dataset

    def generate_sentence_scrambler_dataset(self, data: dict) -> list[dict]:
        """
        Génère le dataset pour le modèle Sentence Scrambler
        Format: {"input": "context|level|sentence", "output": "scrambled_words|||original_sentence"}
        """
        dataset = []

        for context in self.contexts:
            articles = data["articles"].get(context, [])

            for article in articles[:30]:  # Limite
                content = article.get("content", "")
                sentences = self.extract_suitable_sentences(
                    content, min_words=6, max_words=15
                )

                for sentence in sentences[:8]:  # Max 8 phrases par article
                    for level in self.levels:
                        # Ajuster la complexité selon le niveau
                        complexity_factor = self.get_complexity_factor(level)

                        if self.is_sentence_appropriate_for_level(sentence, level):
                            scrambled = self.scramble_sentence(
                                sentence, complexity_factor
                            )

                            dataset.append(
                                {
                                    "input": f"{context}|{level}|{sentence}",
                                    "output": f"{scrambled}|||{sentence}",
                                    "metadata": {
                                        "context": context,
                                        "level": level,
                                        "word_count": len(sentence.split()),
                                        "complexity": complexity_factor,
                                    },
                                }
                            )

        logger.info(f"🔀 {len(dataset)} exemples générés pour Sentence Scrambler")
        return dataset

    def generate_definition_matcher_dataset(self, data: dict) -> list[dict]:
        """
        Génère le dataset pour le modèle Definition Matcher
        Format: {"input": "context|level", "output": "word1,word2,word3|||def1|||def2|||def3|||1,2,3"}
        """
        dataset = []

        for context in self.contexts:
            definitions = data["definitions"].get(context, [])
            words_by_level = defaultdict(list)

            # Organiser par niveau
            for word_data in definitions:
                if word_data.get("has_definition") and word_data.get("definitions"):
                    words_by_level[word_data["level"]].append(word_data)

            for level in self.levels:
                available_words = words_by_level[level]
                if len(available_words) < 3:
                    continue

                # Créer des groupes de 3 mots
                num_groups = min(
                    50, len(available_words) // 3
                )  # Max 50 groupes par niveau

                for _ in range(num_groups):
                    # Sélectionner 3 mots aléatoirement
                    selected_words = random.sample(available_words, 3)

                    words = []
                    definitions = []

                    for word_data in selected_words:
                        words.append(word_data["word"])
                        # Prendre la première définition disponible
                        best_def = self.get_best_definition(word_data["definitions"])
                        definitions.append(best_def)

                    # Mélanger les définitions pour créer l'exercice
                    shuffled_definitions = definitions.copy()
                    random.shuffle(shuffled_definitions)

                    # Trouver les bonnes correspondances
                    correct_matches = []
                    for orig_def in definitions:
                        correct_matches.append(
                            str(shuffled_definitions.index(orig_def) + 1)
                        )

                    dataset.append(
                        {
                            "input": f"{context}|{level}",
                            "output": f"{','.join(words)}|||{'|||'.join(shuffled_definitions)}|||{','.join(correct_matches)}",
                            "metadata": {
                                "context": context,
                                "level": level,
                                "words": words,
                                "original_definitions": definitions,
                            },
                        }
                    )

        logger.info(f"🎯 {len(dataset)} exemples générés pour Definition Matcher")
        return dataset

    def extract_suitable_sentences(
        self, text: str, min_words: int = 8, max_words: int = 25
    ) -> list[str]:
        """Extrait des phrases appropriées pour les exercices"""
        doc = self.nlp(text)
        suitable_sentences = []

        for sent in doc.sents:
            sentence = sent.text.strip()
            word_count = len(sentence.split())

            # Filtres de qualité
            if (
                min_words <= word_count <= max_words
                and not sentence.startswith(("http", "www", "@"))
                and sentence.count(".") <= 2  # Pas trop de phrases complexes
                and not re.search(r"[A-Z]{3,}", sentence)
            ):  # Éviter les acronymes longs

                suitable_sentences.append(sentence)

        return suitable_sentences[:10]  # Limite par texte

    def create_fill_in_blank_variants(
        self, sentence: str, keywords: list[dict], level: str
    ) -> list[dict]:
        """Crée des variantes de texte à trous"""
        variants = []
        doc = self.nlp(sentence)

        # Mots candidats selon le niveau
        difficulty_multiplier = {"beginner": 1, "intermediate": 2, "advanced": 3}[level]
        num_blanks = min(difficulty_multiplier, len(sentence.split()) // 4)

        # Trouver les mots à remplacer
        keyword_words = {kw["word"].lower() for kw in keywords}
        candidate_positions = []

        for i, token in enumerate(doc):
            if (
                token.lemma_.lower() in keyword_words
                and not token.is_stop
                and not token.is_punct
                and len(token.text) > 3
            ):
                candidate_positions.append((i, token.text, token.lemma_.lower()))

        if len(candidate_positions) >= num_blanks:
            # Créer plusieurs variantes
            for _ in range(min(3, len(candidate_positions))):
                selected = random.sample(
                    candidate_positions, min(num_blanks, len(candidate_positions))
                )

                text_with_blanks = sentence
                words_to_fill = []
                target_words = []

                # Remplacer les mots sélectionnés par des trous
                for _pos, word, lemma in selected:
                    text_with_blanks = text_with_blanks.replace(word, "___", 1)
                    words_to_fill.append(word)
                    target_words.append(lemma)

                # Ajouter des mots distracteurs
                distractors = self.get_distractors(target_words, keywords, 2)
                all_options = words_to_fill + distractors
                random.shuffle(all_options)

                variants.append(
                    {
                        "text_with_blanks": text_with_blanks,
                        "words_to_fill": ",".join(all_options),
                        "target_words": target_words,
                    }
                )

        return variants

    def scramble_sentence(self, sentence: str, complexity: float) -> str:
        """Mélange les mots d'une phrase selon la complexité"""
        words = sentence.split()

        if complexity <= 0.5:  # Beginner: mélange simple
            random.shuffle(words)
        elif complexity <= 0.8:  # Intermediate: garder quelques mots en place
            indices_to_shuffle = random.sample(range(len(words)), int(len(words) * 0.7))
            words_to_shuffle = [words[i] for i in indices_to_shuffle]
            random.shuffle(words_to_shuffle)
            for i, idx in enumerate(indices_to_shuffle):
                words[idx] = words_to_shuffle[i]
        else:  # Advanced: mélange complexe avec respect de certaines règles
            # Implémenter une logique plus sophistiquée
            random.shuffle(words)

        return " ".join(words)

    def get_best_definition(self, definitions: list[dict]) -> str:
        """Sélectionne la meilleure définition"""
        if not definitions:
            return "No definition available"

        # Préférer les définitions courtes et claires
        best_def = min(definitions, key=lambda d: len(d.get("definition", "")))
        return best_def.get("definition", "No definition available")

    def get_distractors(
        self, target_words: list[str], keywords: list[dict], num_distractors: int
    ) -> list[str]:
        """Génère des mots distracteurs"""
        distractors = []
        available_words = [
            kw["word"] for kw in keywords if kw["word"] not in target_words
        ]

        if len(available_words) >= num_distractors:
            distractors = random.sample(available_words, num_distractors)

        return distractors

    def get_complexity_factor(self, level: str) -> float:
        """Retourne un facteur de complexité selon le niveau"""
        return {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9}[level]

    def is_sentence_appropriate_for_level(self, sentence: str, level: str) -> bool:
        """Vérifie si une phrase est appropriée pour un niveau"""
        word_count = len(sentence.split())

        if level == "beginner":
            return 6 <= word_count <= 12
        elif level == "intermediate":
            return 8 <= word_count <= 18
        else:  # advanced
            return 10 <= word_count <= 25

    def split_and_save_datasets(self, datasets: dict[str, list[dict]]):
        """Divise et sauvegarde les datasets en train/val/test"""
        for model_name, dataset in datasets.items():
            if not dataset:
                logger.warning(f"⚠️ Dataset vide pour {model_name}")
                continue

            # Division train/val/test (70/15/15)
            train_data, temp_data = train_test_split(
                dataset, test_size=0.3, random_state=42
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=42
            )

            # Créer le dossier pour ce modèle
            model_dir = f"{self.output_dir}/{model_name}"
            os.makedirs(model_dir, exist_ok=True)

            # Sauvegarder les splits
            splits = {"train": train_data, "val": val_data, "test": test_data}

            for split_name, split_data in splits.items():
                output_file = f"{model_dir}/{split_name}.jsonl"
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                logger.info(
                    f"💾 {len(split_data)} exemples sauvegardés dans {output_file}"
                )

    def generate_all_datasets(self):
        """Génère tous les datasets pour les 3 modèles"""
        logger.info("🚀 Génération des datasets d'entraînement pour les modèles d'IA")

        # Charger les données sources
        source_data = self.load_source_data()

        # Générer les datasets pour chaque modèle
        datasets = {
            "fill_in_blank": self.generate_fill_in_blank_dataset(source_data),
            "sentence_scrambler": self.generate_sentence_scrambler_dataset(source_data),
            "definition_matcher": self.generate_definition_matcher_dataset(source_data),
        }

        # Sauvegarder avec splits train/val/test
        self.split_and_save_datasets(datasets)

        # Statistiques finales
        total_examples = sum(len(dataset) for dataset in datasets.values())
        logger.info(f"✅ {total_examples} exemples d'entraînement générés au total")

        for model_name, dataset in datasets.items():
            logger.info(f"   - {model_name}: {len(dataset)} exemples")

        return datasets


def main():
    """Fonction principale"""
    generator = AIDatasetGenerator()
    generator.generate_all_datasets()

    logger.info(
        "🎉 Génération terminée ! Les datasets sont prêts pour l'entraînement des modèles."
    )


if __name__ == "__main__":
    main()
