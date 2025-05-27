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


class FrenchDatasetGenerator:
    """
    Générateur de datasets français pour entraîner les modèles d'IA
    qui créeront les exercices de français pour apprenants anglophones
    """

    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")
        self.contexts = ["it", "work", "travel", "cooking"]
        self.levels = ["beginner", "intermediate", "advanced"]

        # Chemins des données sources françaises
        self.articles_dir = "data/articles_fr"
        self.keywords_dir = "datasets/keywords_fr"
        self.definitions_dir = "datasets/definitions_fr"

        # Chemin de sortie pour les datasets d'entraînement français
        self.output_dir = "datasets/training_fr"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_french_source_data(self) -> dict:
        """Charge toutes les données sources françaises"""
        data = {"articles": {}, "keywords": {}, "definitions": {}}

        for context in self.contexts:
            # Articles français
            articles_file = f"{self.articles_dir}/{context}/articles.jsonl"
            if os.path.exists(articles_file):
                with open(articles_file, encoding="utf-8") as f:
                    data["articles"][context] = [
                        json.loads(line) for line in f if line.strip()
                    ]

            # Keywords français
            keywords_file = f"{self.keywords_dir}/{context}_keywords_fr.json"
            if os.path.exists(keywords_file):
                with open(keywords_file, encoding="utf-8") as f:
                    data["keywords"][context] = json.load(f)

            # Definitions françaises
            definitions_file = f"{self.definitions_dir}/{context}_definitions_fr.json"
            if os.path.exists(definitions_file):
                with open(definitions_file, encoding="utf-8") as f:
                    data["definitions"][context] = json.load(f)

        logger.info("✅ Données sources françaises chargées")
        return data

    def generate_french_fill_in_blank_dataset(self, data: dict) -> list[dict]:
        """
        Génère le dataset français pour le modèle Fill-in-the-Blank
        Format: {"input": "context|level|text", "output": "text_with_blanks|||words_to_fill|||complete_text"}
        """
        dataset = []

        for context in self.contexts:
            articles = data["articles"].get(context, [])
            keywords_by_level = defaultdict(list)

            # Organiser les mots-clés français par niveau
            for kw in data["keywords"].get(context, []):
                keywords_by_level[kw["level"]].append(kw)

            for article in articles[:50]:  # Limite pour éviter trop de données
                content = article.get("content", "")

                # Extraire des phrases françaises appropriées
                sentences = self.extract_french_suitable_sentences(content)

                for sentence in sentences[:5]:  # Max 5 phrases par article
                    for level in self.levels:
                        relevant_words = keywords_by_level[level]
                        if not relevant_words:
                            continue

                        # Créer des variantes françaises avec différents mots supprimés
                        blanked_versions = self.create_french_fill_in_blank_variants(
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
                                        "language": "french",
                                    },
                                }
                            )

        logger.info(f"📝 {len(dataset)} exemples français générés pour Fill-in-the-Blank")
        return dataset

    def generate_french_sentence_scrambler_dataset(self, data: dict) -> list[dict]:
        """
        Génère le dataset français pour le modèle Sentence Scrambler
        Format: {"input": "context|level|sentence", "output": "scrambled_words|||original_sentence"}
        """
        dataset = []

        for context in self.contexts:
            articles = data["articles"].get(context, [])

            for article in articles[:30]:  # Limite
                content = article.get("content", "")
                sentences = self.extract_french_suitable_sentences(
                    content, min_words=6, max_words=18  # Ajusté pour le français
                )

                for sentence in sentences[:8]:  # Max 8 phrases par article
                    for level in self.levels:
                        # Ajuster la complexité selon le niveau
                        complexity_factor = self.get_complexity_factor(level)

                        if self.is_french_sentence_appropriate_for_level(sentence, level):
                            scrambled = self.scramble_french_sentence(
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
                                        "language": "french",
                                    },
                                }
                            )

        logger.info(f"🔀 {len(dataset)} exemples français générés pour Sentence Scrambler")
        return dataset

    def generate_french_definition_matcher_dataset(self, data: dict) -> list[dict]:
        """
        Génère le dataset français pour le modèle Definition Matcher
        Format: {"input": "context|level", "output": "word1,word2,word3|||def1|||def2|||def3|||1,2,3"}
        """
        dataset = []

        for context in self.contexts:
            definitions = data["definitions"].get(context, [])
            words_by_level = defaultdict(list)

            # Organiser par niveau (mots français avec définitions)
            for word_data in definitions:
                if word_data.get("has_definition") and word_data.get("definitions"):
                    words_by_level[word_data["level"]].append(word_data)

            for level in self.levels:
                available_words = words_by_level[level]
                if len(available_words) < 3:
                    continue

                # Créer des groupes de 3 mots français
                num_groups = min(
                    50, len(available_words) // 3
                )  # Max 50 groupes par niveau

                for _ in range(num_groups):
                    # Sélectionner 3 mots français aléatoirement
                    selected_words = random.sample(available_words, 3)

                    words = []
                    definitions = []

                    for word_data in selected_words:
                        words.append(word_data["word"])
                        # Prendre la meilleure définition française disponible
                        best_def = self.get_best_french_definition(word_data["definitions"])
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
                                "language": "french",
                            },
                        }
                    )

        logger.info(f"🎯 {len(dataset)} exemples français générés pour Definition Matcher")
        return dataset

    def extract_french_suitable_sentences(
            self, text: str, min_words: int = 8, max_words: int = 30
    ) -> list[str]:
        """Extrait des phrases françaises appropriées pour les exercices"""
        doc = self.nlp(text)
        suitable_sentences = []

        for sent in doc.sents:
            sentence = sent.text.strip()
            word_count = len(sentence.split())

            # Filtres de qualité pour le français
            if (
                    min_words <= word_count <= max_words
                    and not sentence.startswith(("http", "www", "@"))
                    and sentence.count(".") <= 2  # Pas trop de phrases complexes
                    and not re.search(r"[A-Z]{3,}", sentence)  # Éviter les acronymes longs
                    and self.is_proper_french_sentence(sentence)
            ):
                suitable_sentences.append(sentence)

        return suitable_sentences[:10]  # Limite par texte

    def is_proper_french_sentence(self, sentence: str) -> bool:
        """Vérifie qu'une phrase est correctement formée en français"""
        # Vérifications basiques pour le français
        if len(sentence) < 10:
            return False

        # Doit contenir au moins un verbe ou nom (heuristique simple)
        doc = self.nlp(sentence)
        has_verb_or_noun = any(token.pos_ in ["VERB", "NOUN"] for token in doc)

        # Pas trop de caractères spéciaux
        special_char_ratio = len(re.findall(r'[^a-zA-ZÀ-ÿ\s\.,!?;:]', sentence)) / len(sentence)

        return has_verb_or_noun and special_char_ratio < 0.1

    def create_french_fill_in_blank_variants(
            self, sentence: str, keywords: list[dict], level: str
    ) -> list[dict]:
        """Crée des variantes françaises de texte à trous"""
        variants = []
        doc = self.nlp(sentence)

        # Mots candidats selon le niveau (ajusté pour le français)
        difficulty_multiplier = {"beginner": 1, "intermediate": 2, "advanced": 3}[level]
        num_blanks = min(difficulty_multiplier, len(sentence.split()) // 5)

        # Trouver les mots français à remplacer
        keyword_words = {kw["word"].lower() for kw in keywords}
        candidate_positions = []

        for i, token in enumerate(doc):
            if (
                    token.lemma_.lower() in keyword_words
                    and not token.is_stop
                    and not token.is_punct
                    and len(token.text) > 3
                    and token.pos_ in ["NOUN", "VERB", "ADJ"]  # Types de mots pertinents
            ):
                candidate_positions.append((i, token.text, token.lemma_.lower()))

        if len(candidate_positions) >= num_blanks:
            # Créer plusieurs variantes françaises
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

                # Ajouter des mots distracteurs français
                distractors = self.get_french_distractors(target_words, keywords, 3)
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

    def scramble_french_sentence(self, sentence: str, complexity: float) -> str:
        """Mélange les mots d'une phrase française selon la complexité"""
        words = sentence.split()

        if complexity <= 0.5:  # Beginner: mélange simple
            random.shuffle(words)
        elif complexity <= 0.8:  # Intermediate: garder quelques mots en place
            indices_to_shuffle = random.sample(range(len(words)), int(len(words) * 0.7))
            words_to_shuffle = [words[i] for i in indices_to_shuffle]
            random.shuffle(words_to_shuffle)
            for i, idx in enumerate(indices_to_shuffle):
                words[idx] = words_to_shuffle[i]
        else:  # Advanced: mélange complexe avec respect des groupes nominaux
            # Pour le français, essayer de respecter certains groupes
            doc = self.nlp(sentence)

            # Identifier les groupes à ne pas séparer (déterminant + nom)
            protected_groups = []
            for i, token in enumerate(doc):
                if token.pos_ == "DET" and i + 1 < len(doc) and doc[i + 1].pos_ == "NOUN":
                    protected_groups.append((i, i + 1))

            # Mélanger en respectant ces groupes (version simplifiée)
            random.shuffle(words)

        return " ".join(words)

    def get_best_french_definition(self, definitions: list[dict]) -> str:
        """Sélectionne la meilleure définition française"""
        if not definitions:
            return "Définition non disponible"

        # Préférer les définitions courtes et claires en français
        valid_definitions = [
            d for d in definitions
            if d.get("definition") and len(d["definition"].strip()) > 10
        ]

        if not valid_definitions:
            return "Définition non disponible"

        # Choisir la définition la plus courte et la plus claire
        best_def = min(
            valid_definitions,
            key=lambda d: len(d.get("definition", "")) + (50 if "(" in d.get("definition", "") else 0)
        )

        definition_text = best_def.get("definition", "Définition non disponible")

        # Nettoyer la définition
        definition_text = definition_text.strip()
        if not definition_text.endswith('.'):
            definition_text += '.'

        return definition_text

    def get_french_distractors(
            self, target_words: list[str], keywords: list[dict], num_distractors: int
    ) -> list[str]:
        """Génère des mots distracteurs français"""
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

    def is_french_sentence_appropriate_for_level(self, sentence: str, level: str) -> bool:
        """Vérifie si une phrase française est appropriée pour un niveau"""
        word_count = len(sentence.split())

        # Ajusté pour le français (phrases généralement plus longues)
        if level == "beginner":
            return 6 <= word_count <= 14
        elif level == "intermediate":
            return 10 <= word_count <= 20
        else:  # advanced
            return 12 <= word_count <= 30

    def split_and_save_french_datasets(self, datasets: dict[str, list[dict]]):
        """Divise et sauvegarde les datasets français en train/val/test"""
        for model_name, dataset in datasets.items():
            if not dataset:
                logger.warning(f"⚠️ Dataset français vide pour {model_name}")
                continue

            # Division train/val/test (70/15/15)
            train_data, temp_data = train_test_split(
                dataset, test_size=0.3, random_state=42, shuffle=True
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=42, shuffle=True
            )

            # Créer le dossier pour ce modèle français
            model_dir = f"{self.output_dir}/{model_name}"
            os.makedirs(model_dir, exist_ok=True)

            # Sauvegarder les splits français
            splits = {"train": train_data, "val": val_data, "test": test_data}

            for split_name, split_data in splits.items():
                output_file = f"{model_dir}/{split_name}.jsonl"
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                logger.info(
                    f"💾 {len(split_data)} exemples français sauvegardés dans {output_file}"
                )

    def generate_all_french_datasets(self):
        """Génère tous les datasets français pour les 3 modèles"""
        logger.info("🚀 Génération des datasets d'entraînement français pour les modèles d'IA")

        # Charger les données sources françaises
        source_data = self.load_french_source_data()

        # Générer les datasets français pour chaque modèle
        datasets = {
            "fill_in_blank": self.generate_french_fill_in_blank_dataset(source_data),
            "sentence_scrambler": self.generate_french_sentence_scrambler_dataset(source_data),
            "definition_matcher": self.generate_french_definition_matcher_dataset(source_data),
        }

        # Sauvegarder avec splits train/val/test
        self.split_and_save_french_datasets(datasets)

        # Statistiques finales
        total_examples = sum(len(dataset) for dataset in datasets.values())
        logger.info(f"✅ {total_examples} exemples d'entraînement français générés au total")

        for model_name, dataset in datasets.items():
            logger.info(f"   - {model_name}: {len(dataset)} exemples français")

        return datasets


def main():
    """Fonction principale pour générer les datasets français"""
    generator = FrenchDatasetGenerator()
    generator.generate_all_french_datasets()

    logger.info(
        "🎉 Génération française terminée ! Les datasets sont prêts pour l'entraînement des modèles."
    )


if __name__ == "__main__":
    main()
