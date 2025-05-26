import json
import logging
import os
import re
from collections import Counter, defaultdict

import spacy
from textstat import flesch_reading_ease

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialise l'extracteur avec spaCy
        Args:
            model_name: Modèle spaCy à utiliser (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✅ Modèle spaCy '{model_name}' chargé avec succès")
        except OSError:
            logger.error(f"❌ Modèle spaCy '{model_name}' non trouvé")
            logger.info("💡 Installez-le avec: python -m spacy download en_core_web_sm")
            raise

        # Configuration pour l'extraction
        self.min_word_length = 3
        self.max_word_length = 20
        self.min_frequency = 2

        # POS tags à conserver pour les mots-clés
        self.valid_pos = {"NOUN", "PROPN", "ADJ", "VERB"}

        # Mots-outils à ignorer même s'ils ne sont pas dans stop_words
        self.blacklist = {
            "say",
            "get",
            "go",
            "come",
            "take",
            "make",
            "see",
            "know",
            "think",
            "look",
            "want",
            "give",
            "use",
            "find",
            "tell",
            "ask",
            "work",
            "seem",
            "year",
            "time",
            "day",
            "way",
            "new",
            "first",
            "last",
            "long",
            "great",
            "little",
            "own",
            "other",
            "old",
            "right",
            "big",
            "high",
            "small",
            "large",
            "good",
            "bad",
            "people",
            "man",
            "woman",
            "child",
            "person",
        }

    def clean_text(self, text: str) -> str:
        """Nettoie le texte avant traitement"""
        # Supprimer les URLs
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+", "", text)
        # Supprimer les emails
        text = re.sub(r"\S+@\S+", "", text)
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r"[^\w\s\.\!\?\,\;\:]", " ", text)
        # Normaliser les espaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_keywords_from_text(
        self, text: str, top_k: int = 30
    ) -> list[dict[str, any]]:
        """
        Extrait les mots-clés d'un texte avec spaCy
        Args:
            text: Texte à analyser
            top_k: Nombre maximum de mots-clés à retourner
        Returns:
            Liste de dictionnaires avec mot, fréquence, POS, etc.
        """
        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)

        # Extraire les tokens pertinents
        word_info = defaultdict(
            lambda: {"count": 0, "pos_tags": set(), "lemma": "", "contexts": []}
        )

        for token in doc:
            # Filtres de base
            if (
                token.is_stop
                or token.is_punct
                or token.is_space
                or len(token.text) < self.min_word_length
                or len(token.text) > self.max_word_length
                or token.pos_ not in self.valid_pos
            ):
                continue

            # Lemme en minuscules
            lemma = token.lemma_.lower()

            # Ignorer les mots de la blacklist
            if lemma in self.blacklist:
                continue

            # Ignorer les mots qui sont principalement des chiffres
            if re.match(r"^[\d\.\,]+$", lemma):
                continue

            # Collecter les informations
            word_info[lemma]["count"] += 1
            word_info[lemma]["pos_tags"].add(token.pos_)
            word_info[lemma]["lemma"] = lemma

            # Contexte (phrase contenant le mot)
            sentence = token.sent.text.strip()
            if len(sentence) > 20 and len(word_info[lemma]["contexts"]) < 3:
                word_info[lemma]["contexts"].append(sentence)

        # Filtrer par fréquence minimale et convertir en liste
        keywords = []
        for word, info in word_info.items():
            if info["count"] >= self.min_frequency:
                keywords.append(
                    {
                        "word": word,
                        "frequency": info["count"],
                        "pos_tags": list(info["pos_tags"]),
                        "contexts": info["contexts"],
                        "importance_score": self.calculate_importance_score(
                            word, info, len(doc)
                        ),
                    }
                )

        # Trier par score d'importance
        keywords.sort(key=lambda x: x["importance_score"], reverse=True)

        return keywords[:top_k]

    def calculate_importance_score(
        self, word: str, info: dict, doc_length: int
    ) -> float:
        """
        Calcule un score d'importance pour un mot-clé
        Combine fréquence, longueur du mot, et types de POS
        """
        # Score de base = fréquence normalisée
        base_score = info["count"] / doc_length * 1000

        # Bonus pour les mots plus longs (plus spécifiques)
        length_bonus = min(len(word) / 10, 1.0)

        # Bonus selon le type de POS
        pos_bonus = 0
        if "PROPN" in info["pos_tags"]:  # Noms propres = très importants
            pos_bonus = 0.5
        elif "NOUN" in info["pos_tags"]:  # Noms = importants
            pos_bonus = 0.3
        elif "ADJ" in info["pos_tags"]:  # Adjectifs = moyennement importants
            pos_bonus = 0.2

        return base_score + length_bonus + pos_bonus

    def classify_difficulty_level(
        self, word: str, context: str, article_text: str
    ) -> str:
        """
        Classifie le niveau de difficulté d'un mot
        Args:
            word: Le mot à classifier
            context: Contexte (it, work, travel)
            article_text: Texte complet de l'article
        Returns:
            'beginner', 'intermediate', ou 'advanced'
        """
        # 1. Longueur du mot (heuristique simple)
        if len(word) <= 4:
            length_score = 1  # facile
        elif len(word) <= 8:
            length_score = 2  # moyen
        else:
            length_score = 3  # difficile

        # 2. Complexité de l'article (Flesch Reading Ease)
        try:
            flesch_score = flesch_reading_ease(article_text)
            if flesch_score >= 60:  # Facile à lire
                article_difficulty = 1
            elif flesch_score >= 30:  # Moyennement difficile
                article_difficulty = 2
            else:  # Difficile
                article_difficulty = 3
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul du Flesch score: {e}")
            article_difficulty = 2  # Par défaut

        # 3. Spécificité du domaine (mots techniques vs communs)
        domain_specific_patterns = {
            "it": [
                r".*tech.*",
                r".*cyber.*",
                r".*soft.*",
                r".*data.*",
                r".*api.*",
                r".*dev.*",
            ],
            "work": [
                r".*manage.*",
                r".*business.*",
                r".*strategy.*",
                r".*market.*",
                r".*leader.*",
            ],
            "travel": [
                r".*destination.*",
                r".*culture.*",
                r".*adventure.*",
                r".*journey.*",
            ],
        }

        domain_score = 1  # Par défaut
        if context in domain_specific_patterns:
            for pattern in domain_specific_patterns[context]:
                if re.match(pattern, word.lower()):
                    domain_score = 3  # Mot spécialisé = plus difficile
                    break

        # 4. Score final
        final_score = (length_score + article_difficulty + domain_score) / 3

        if final_score <= 1.5:
            return "beginner"
        elif final_score <= 2.5:
            return "intermediate"
        else:
            return "advanced"

    def process_articles_file(self, articles_file: str, context: str) -> list[dict]:
        """
        Traite un fichier d'articles JSONL et extrait les mots-clés
        Args:
            articles_file: Chemin vers le fichier JSONL
            context: Contexte (it, work, travel)
        Returns:
            Liste des mots-clés extraits avec métadonnées
        """
        logger.info(f"🔍 Traitement du fichier: {articles_file}")

        if not os.path.exists(articles_file):
            logger.error(f"❌ Fichier non trouvé: {articles_file}")
            return []

        all_keywords = []
        word_global_count = Counter()

        # Première passe: compter tous les mots pour identifier les plus fréquents
        with open(articles_file, encoding="utf-8") as f:
            articles = [json.loads(line) for line in f if line.strip()]

        logger.info(f"📄 {len(articles)} articles à traiter")

        # Traiter chaque article
        for i, article in enumerate(articles):
            try:
                title = article.get("title", "")
                content = article.get("content", "")
                full_text = f"{title}\n\n{content}"

                if len(full_text.split()) < 50:  # Ignorer les articles trop courts
                    continue

                # Extraire les mots-clés de cet article
                keywords = self.extract_keywords_from_text(full_text, top_k=20)

                # Enrichir avec métadonnées
                for kw in keywords:
                    difficulty = self.classify_difficulty_level(
                        kw["word"], context, full_text
                    )

                    all_keywords.append(
                        {
                            "word": kw["word"],
                            "context": context,
                            "level": difficulty,
                            "frequency_in_article": kw["frequency"],
                            "importance_score": kw["importance_score"],
                            "pos_tags": kw["pos_tags"],
                            "contexts": kw["contexts"],
                            "source_url": article.get("url", ""),
                            "source_title": title,
                            "article_index": i,
                        }
                    )

                    word_global_count[kw["word"]] += kw["frequency"]

                if (i + 1) % 10 == 0:
                    logger.info(f"✅ {i + 1}/{len(articles)} articles traités")

            except Exception as e:
                logger.error(f"❌ Erreur article {i}: {e}")
                continue

        # Ajouter la fréquence globale et filtrer les doublons
        final_keywords = {}
        for kw in all_keywords:
            word = kw["word"]
            kw["global_frequency"] = word_global_count[word]

            # Garder le meilleur exemple de chaque mot
            if (
                word not in final_keywords
                or kw["importance_score"] > final_keywords[word]["importance_score"]
            ):
                final_keywords[word] = kw

        # Convertir en liste et trier
        result = list(final_keywords.values())
        result.sort(key=lambda x: x["global_frequency"], reverse=True)

        logger.info(f"🎯 {len(result)} mots-clés uniques extraits pour '{context}'")
        return result

    def save_keywords(self, keywords: list[dict], output_file: str):
        """Sauvegarde les mots-clés dans un fichier JSON"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 {len(keywords)} mots-clés sauvegardés dans {output_file}")


def main():
    """Fonction principale pour tester l'extracteur"""
    extractor = KeywordExtractor()

    # Répertoires
    articles_dir = "data/articles"
    keywords_dir = "datasets/keywords"
    os.makedirs(keywords_dir, exist_ok=True)

    # Traiter chaque contexte
    contexts = ["it", "work", "travel"]

    for context in contexts:
        articles_file = f"{articles_dir}/{context}/articles.jsonl"

        if os.path.exists(articles_file):
            logger.info(f"🚀 Extraction des mots-clés pour '{context}'")

            keywords = extractor.process_articles_file(articles_file, context)

            if keywords:
                output_file = f"{keywords_dir}/{context}_keywords.json"
                extractor.save_keywords(keywords, output_file)

                # Statistiques
                levels = Counter(kw["level"] for kw in keywords)
                logger.info(
                    f"📊 Distribution des niveaux pour '{context}': {dict(levels)}"
                )
            else:
                logger.warning(f"⚠️ Aucun mot-clé extrait pour '{context}'")
        else:
            logger.warning(f"⚠️ Fichier d'articles non trouvé: {articles_file}")

    logger.info("🎉 Extraction terminée pour tous les contextes")


if __name__ == "__main__":
    main()
