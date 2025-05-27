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


class FrenchKeywordExtractor:
    def __init__(self, model_name="fr_core_news_sm"):
        """
        Initialise l'extracteur avec spaCy français
        Args:
            model_name: Modèle spaCy français (fr_core_news_sm, fr_core_news_md, fr_core_news_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✅ Modèle spaCy français '{model_name}' chargé avec succès")
        except OSError:
            logger.error(f"❌ Modèle spaCy français '{model_name}' non trouvé")
            logger.info("💡 Installez-le avec: python -m spacy download fr_core_news_sm")
            raise

        # Configuration pour l'extraction française
        self.min_word_length = 3
        self.max_word_length = 25  # Français a des mots plus longs
        self.min_frequency = 2

        # POS tags français à conserver pour les mots-clés
        self.valid_pos = {"NOUN", "PROPN", "ADJ", "VERB"}

        # Mots-outils français à ignorer (stop words personnalisés)
        self.french_blacklist = {
            # Verbes très courants
            "être", "avoir", "faire", "dire", "aller", "voir", "savoir", "prendre",
            "venir", "vouloir", "pouvoir", "falloir", "devoir", "croire", "trouver",
            "donner", "parler", "aimer", "porter", "laisser", "entendre", "demander",
            "rester", "passer", "arriver", "entrer", "monter", "sortir", "partir",
            "tenir", "finir", "jouer", "tourner", "servir", "ouvrir", "mettre",

            # Noms très génériques
            "année", "temps", "jour", "moment", "façon", "chose", "cas", "part",
            "lieu", "place", "fois", "point", "nombre", "partie", "côté", "main",
            "gens", "homme", "femme", "enfant", "personne", "monde", "pays",
            "ville", "maison", "école", "travail", "problème", "question", "vie",

            # Adjectifs très courants
            "nouveau", "premier", "dernier", "long", "grand", "petit", "autre",
            "vieux", "beau", "gros", "jeune", "bon", "mauvais", "français",
            "national", "international", "public", "privé", "social", "politique",
            "économique", "important", "différent", "possible", "certain",

            # Déterminants et pronoms non captés par spaCy
            "tout", "tous", "toute", "toutes", "chaque", "plusieurs", "quelque",
            "aucun", "même", "tel", "cette", "cela", "celui", "celle",
        }

    def clean_french_text(self, text: str) -> str:
        """Nettoie le texte français avant traitement"""
        # Supprimer les URLs
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+", "", text)
        # Supprimer les emails
        text = re.sub(r"\S+@\S+", "", text)
        # Supprimer les caractères spéciaux mais garder les accents français
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\àâäéèêëïîôöùûüÿç]", " ", text)
        # Normaliser les espaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_french_keywords(self, text: str, top_k: int = 30) -> list[dict[str, any]]:
        """
        Extrait les mots-clés français d'un texte avec spaCy
        """
        cleaned_text = self.clean_french_text(text)
        doc = self.nlp(cleaned_text)

        # Extraire les tokens pertinents
        word_info = defaultdict(
            lambda: {"count": 0, "pos_tags": set(), "lemma": "", "contexts": []}
        )

        for token in doc:
            # Filtres de base pour le français
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

            # Ignorer les mots de la blacklist française
            if lemma in self.french_blacklist:
                continue

            # Ignorer les mots principalement numériques
            if re.match(r"^[\d\.\,]+$", lemma):
                continue

            # Ignorer les mots trop courts après lemmatisation
            if len(lemma) < 3:
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
        french_keywords = []
        for word, info in word_info.items():
            if info["count"] >= self.min_frequency:
                french_keywords.append(
                    {
                        "word": word,
                        "frequency": info["count"],
                        "pos_tags": list(info["pos_tags"]),
                        "contexts": info["contexts"],
                        "importance_score": self.calculate_french_importance_score(
                            word, info, len(doc)
                        ),
                    }
                )

        # Trier par score d'importance
        french_keywords.sort(key=lambda x: x["importance_score"], reverse=True)

        return french_keywords[:top_k]

    def calculate_french_importance_score(
            self, word: str, info: dict, doc_length: int
    ) -> float:
        """
        Calcule un score d'importance pour un mot-clé français
        """
        # Score de base = fréquence normalisée
        base_score = info["count"] / doc_length * 1000

        # Bonus pour les mots plus longs (plus spécifiques en français)
        length_bonus = min(len(word) / 12, 1.0)  # Ajusté pour le français

        # Bonus selon le type de POS en français
        pos_bonus = 0
        if "PROPN" in info["pos_tags"]:  # Noms propres = très importants
            pos_bonus = 0.6
        elif "NOUN" in info["pos_tags"]:  # Noms = importants
            pos_bonus = 0.4
        elif "ADJ" in info["pos_tags"]:  # Adjectifs = moyennement importants
            pos_bonus = 0.3
        elif "VERB" in info["pos_tags"]:  # Verbes = moins importants
            pos_bonus = 0.2

        # Bonus pour les mots avec suffixes français complexes
        complex_suffixes = [
            "tion", "sion", "isme", "ique", "eur", "euse", "teur", "trice",
            "ance", "ence", "ité", "ment", "able", "ible"
        ]
        suffix_bonus = 0.2 if any(word.endswith(suffix) for suffix in complex_suffixes) else 0

        return base_score + length_bonus + pos_bonus + suffix_bonus

    def classify_french_difficulty_level(
            self, word: str, context: str, article_text: str
    ) -> str:
        """
        Classifie le niveau de difficulté d'un mot français pour apprenants anglophones
        """
        # 1. Longueur du mot (français a des mots plus longs en moyenne)
        if len(word) <= 5:
            length_score = 1
        elif len(word) <= 10:
            length_score = 2
        else:
            length_score = 3

        # 2. Complexité morphologique française
        complex_endings = [
            "tion", "sion", "ment", "isme", "ique", "eur", "euse",
            "teur", "trice", "ance", "ence", "ité", "able", "ible"
        ]

        morphology_score = 1
        for ending in complex_endings:
            if word.endswith(ending):
                morphology_score = 3
                break

        # 3. Complexité de l'article (Flesch Reading Ease adapté au français)
        try:
            flesch_score = flesch_reading_ease(article_text)
            # Ajustement pour le français (généralement plus bas qu'en anglais)
            if flesch_score >= 50:  # Facile à lire
                article_difficulty = 1
            elif flesch_score >= 20:  # Moyennement difficile
                article_difficulty = 2
            else:  # Difficile
                article_difficulty = 3
        except Exception as e:
            logger.error(f"❌ Erreur calcul Flesch français: {e}")
            article_difficulty = 2

        # 4. Spécificité du domaine français
        domain_patterns = {
            "it": [
                r".*logiciel.*", r".*numérique.*", r".*informatique.*",
                r".*technolog.*", r".*algorithme.*", r".*données.*",
                r".*développ.*", r".*program.*", r".*système.*"
            ],
            "work": [
                r".*entreprise.*", r".*management.*", r".*stratégie.*",
                r".*commercial.*", r".*économique.*", r".*professionnel.*",
                r".*gestio.*", r".*direction.*", r".*business.*"
            ],
            "travel": [
                r".*destination.*", r".*tourisme.*", r".*voyage.*",
                r".*culture.*", r".*patrimoine.*", r".*découverte.*",
                r".*aventure.*", r".*exploration.*"
            ],
            "cooking": [
                r".*cuisine.*", r".*gastronomie.*", r".*ingrédient.*",
                r".*recette.*", r".*culinaire.*", r".*chef.*",
                r".*plat.*", r".*saveur.*", r".*préparation.*"
            ]
        }

        domain_score = 1
        if context in domain_patterns:
            for pattern in domain_patterns[context]:
                if re.match(pattern, word.lower()):
                    domain_score = 3
                    break

        # 5. Score final
        final_score = (length_score + morphology_score + article_difficulty + domain_score) / 4

        if final_score <= 1.5:
            return "beginner"
        elif final_score <= 2.5:
            return "intermediate"
        else:
            return "advanced"

    def process_french_articles_file(self, articles_file: str, context: str) -> list[dict]:
        """
        Traite un fichier d'articles français JSONL et extrait les mots-clés
        """
        logger.info(f"🔍 Traitement du fichier français: {articles_file}")

        if not os.path.exists(articles_file):
            logger.error(f"❌ Fichier non trouvé: {articles_file}")
            return []

        all_keywords = []
        word_global_count = Counter()

        # Charger les articles français
        with open(articles_file, encoding="utf-8") as f:
            articles = [json.loads(line) for line in f if line.strip()]

        logger.info(f"📄 {len(articles)} articles français à traiter")

        # Traiter chaque article
        for i, article in enumerate(articles):
            try:
                title = article.get("title", "")
                content = article.get("content", "")
                full_text = f"{title}\n\n{content}"

                # Ignorer les articles trop courts (français nécessite plus de mots)
                if len(full_text.split()) < 100:
                    continue

                # Extraire les mots-clés français
                keywords = self.extract_french_keywords(full_text, top_k=25)

                # Enrichir avec métadonnées françaises
                for kw in keywords:
                    difficulty = self.classify_french_difficulty_level(
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
                            "language": "french",
                        }
                    )

                    word_global_count[kw["word"]] += kw["frequency"]

                if (i + 1) % 10 == 0:
                    logger.info(f"✅ {i + 1}/{len(articles)} articles français traités")

            except Exception as e:
                logger.error(f"❌ Erreur article français {i}: {e}")
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

        logger.info(f"🎯 {len(result)} mots-clés français uniques extraits pour '{context}'")
        return result

    def save_french_keywords(self, keywords: list[dict], output_file: str):
        """Sauvegarde les mots-clés français dans un fichier JSON"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 {len(keywords)} mots-clés français sauvegardés dans {output_file}")


def main():
    """Fonction principale pour extraire les mots-clés français"""
    extractor = FrenchKeywordExtractor()

    # Répertoires français
    articles_dir = "data/articles_fr"
    keywords_dir = "datasets/keywords_fr"
    os.makedirs(keywords_dir, exist_ok=True)

    # Traiter chaque contexte français
    contexts = ["it", "work", "travel", "cooking"]

    for context in contexts:
        articles_file = f"{articles_dir}/{context}/articles.jsonl"

        if os.path.exists(articles_file):
            logger.info(f"🚀 Extraction des mots-clés français pour '{context}'")

            keywords = extractor.process_french_articles_file(articles_file, context)

            if keywords:
                output_file = f"{keywords_dir}/{context}_keywords_fr.json"
                extractor.save_french_keywords(keywords, output_file)

                # Statistiques par niveau
                levels = Counter(kw["level"] for kw in keywords)
                logger.info(
                    f"📊 Distribution des niveaux français pour '{context}': {dict(levels)}"
                )
            else:
                logger.warning(f"⚠️ Aucun mot-clé français extrait pour '{context}'")
        else:
            logger.warning(f"⚠️ Fichier d'articles français non trouvé: {articles_file}")

    logger.info("🎉 Extraction française terminée pour tous les contextes")


if __name__ == "__main__":
    main()
