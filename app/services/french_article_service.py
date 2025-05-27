import json
import logging
import os
import random
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sources françaises par contexte
FRENCH_SOURCES = {
    "it": [
        "https://www.journaldunet.com/",
        "https://www.01net.com/",
        "https://www.numerama.com/",
        "https://www.lemonde.fr/pixels/",
        "https://www.usine-digitale.fr/",
        "https://www.silicon.fr/",
    ],
    "work": [
        "https://www.challenges.fr/",
        "https://www.lesechos.fr/",
        "https://www.capital.fr/",
        "https://business.lesechos.fr/",
        "https://www.hbrfrance.fr/",
        "https://start.lesechos.fr/",
    ],
    "travel": [
        "https://www.routard.com/",
        "https://www.geo.fr/",
        "https://www.petitfute.com/",
        "https://www.partir.com/",
        "https://www.lonelyplanet.fr/",
        "https://www.voyageurs-du-net.com/",
    ],
    "cooking": [
        "https://www.marmiton.org/",
        "https://www.750g.com/",
        "https://www.cuisineaz.com/",
        "https://madame.lefigaro.fr/cuisine/",
        "https://www.cuisineactuelle.fr/",
        "https://www.ptitchef.com/",
    ],
}

OUTPUT_DIR = "data/articles_fr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


class FrenchArticleScraper:
    def __init__(self, max_articles_per_source: int = 50, delay_range: tuple = (1, 3)):
        self.max_articles_per_source = max_articles_per_source
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_html(self, url: str) -> str:
        """Récupère le contenu HTML d'une URL avec gestion d'erreurs améliorée"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors du fetch de {url}: {e}")
            return ""

    def extract_article_links(self, base_url: str, soup: BeautifulSoup) -> list[str]:
        """Extrait les liens d'articles français"""
        links = set()
        domain = urlparse(base_url).netloc

        # Sélecteurs CSS adaptés aux sites français
        selectors = [
            'a[href*="/article"]',
            'a[href*="/actualite"]',
            'a[href*="/news"]',
            'a[href*="/actu"]',
            'a[href*="/recette"]',  # Pour les sites de cuisine
            'a[href*="/voyage"]',   # Pour les sites de voyage
            'a[href*="/tech"]',     # Pour les sites tech
            "article a",
            ".article-title a",
            ".entry-title a",
            ".post-title a",
            "h2 a",
            "h3 a",
            ".titre a",
            ".headline a",
        ]

        for selector in selectors:
            for link in soup.select(selector):
                href = link.get("href", "")
                if href:
                    full_url = urljoin(base_url, href)
                    if domain in full_url and self.is_french_article_url(full_url):
                        links.add(full_url)
                        if len(links) >= self.max_articles_per_source:
                            break
            if len(links) >= self.max_articles_per_source:
                break

        # Fallback: tous les liens si pas assez trouvés
        if len(links) < 10:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin(base_url, href)
                if domain in full_url and self.is_french_article_url(full_url):
                    links.add(full_url)
                    if len(links) >= self.max_articles_per_source:
                        break

        return list(links)

    def is_french_article_url(self, url: str) -> bool:
        """Détermine si une URL semble être un article français"""
        url_lower = url.lower()

        # URLs à éviter
        avoid_patterns = [
            "/tag/", "/tags/", "/category/", "/categorie/", "/author/", "/auteur/",
            "/page/", "/recherche/", "/search/", "/login", "/connexion",
            "/register", "/inscription", "/contact", "/a-propos", "/about",
            "/mentions-legales", "/cgu", "/privacy", "/confidentialite",
            ".pdf", ".jpg", ".png", ".gif", ".mp4", ".zip",
            "#", "javascript:", "mailto:", "tel:",
            "/newsletter", "/rss", "/feed"
        ]

        if any(pattern in url_lower for pattern in avoid_patterns):
            return False

        # URLs probablement intéressantes (mots-clés français)
        good_patterns = [
            "/article", "/actualite", "/actu", "/news", "/info",
            "/recette", "/cuisine", "/voyage", "/destination",
            "/tech", "/numerique", "/digital", "/innovation",
            "/business", "/economie", "/entreprise", "/management",
            "/2024/", "/2025/",
            "/comment-", "/pourquoi-", "/que-", "/qui-",
        ]

        return (
                any(pattern in url_lower for pattern in good_patterns)
                or len(url.split("/")) >= 4
        )

    def extract_article_content(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extrait le titre et le contenu d'un article français"""
        # Extraction du titre
        title = ""
        title_selectors = [
            "h1",
            ".entry-title",
            ".post-title",
            ".article-title",
            ".titre",
            ".headline",
            ".title",
            "title",
        ]
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                break

        # Extraction du contenu
        content = ""

        # Supprimer les éléments indésirables
        for element in soup([
            "script", "style", "nav", "header", "footer", "aside",
            "advertisement", ".pub", ".publicite", ".ad", ".ads",
            ".newsletter", ".social", ".partage", ".share",
            ".comments", ".commentaires", ".comment-form"
        ]):
            element.decompose()

        # Chercher le contenu principal (sélecteurs français)
        content_selectors = [
            ".entry-content",
            ".post-content",
            ".article-content",
            ".contenu-article",
            ".article-body",
            ".content",
            ".contenu",
            "article",
            ".story-body",
            ".texte-article",
            "main",
            ".main-content",
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                paragraphs = content_elem.find_all("p")
                if paragraphs:
                    content = "\n".join(
                        p.get_text().strip() for p in paragraphs if p.get_text().strip()
                    )
                    break

        # Fallback: tous les paragraphes
        if not content:
            paragraphs = soup.find_all("p")
            content = "\n".join(
                p.get_text().strip() for p in paragraphs if p.get_text().strip()
            )

        return {"title": title, "content": content.strip()}

    def scrape_source(self, context: str, source_url: str) -> list[dict]:
        """Scrape une source française spécifique"""
        logger.info(f"🔍 Scraping {source_url} pour le contexte français '{context}'")

        # Récupérer la page principale
        html = self.fetch_html(source_url)
        if not html:
            logger.warning(f"❌ Impossible de récupérer {source_url}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        article_links = self.extract_article_links(source_url, soup)

        logger.info(f"📄 {len(article_links)} liens d'articles français trouvés")

        articles = []
        for i, link in enumerate(article_links):
            try:
                # Délai aléatoire pour éviter le rate limiting
                time.sleep(random.uniform(*self.delay_range))

                article_html = self.fetch_html(link)
                if not article_html:
                    continue

                article_soup = BeautifulSoup(article_html, "html.parser")
                article_data = self.extract_article_content(article_soup)

                # Filtrer les articles trop courts (français nécessite plus de mots)
                if len(article_data["content"].split()) < 150:
                    logger.debug(f"⚠️ Article français trop court ignoré: {link}")
                    continue

                # Vérifier que c'est bien du français (heuristique simple)
                if not self.is_likely_french_text(article_data["content"]):
                    logger.debug(f"⚠️ Article probablement pas en français: {link}")
                    continue

                articles.append(
                    {
                        "context": context,
                        "source": urlparse(source_url).netloc,
                        "url": link,
                        "title": article_data["title"],
                        "content": article_data["content"],
                        "language": "french",
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                logger.info(
                    f"✅ Article français {i+1}/{len(article_links)} traité: {article_data['title'][:50]}..."
                )

            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {link}: {e}")
                continue

        logger.info(f"🎯 {len(articles)} articles français valides récupérés de {source_url}")
        return articles

    def is_likely_french_text(self, text: str) -> bool:
        """Heuristique simple pour détecter du texte français"""
        if not text or len(text) < 100:
            return False

        # Mots français courants
        french_words = [
            "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir",
            "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne",
            "se", "pas", "tout", "plus", "par", "grand", "comme", "mais",
            "dans", "cette", "des", "les", "du", "la", "leur", "ses",
            "français", "france", "aussi", "très", "nous", "vous", "ils"
        ]

        # Compter les mots français
        words = text.lower().split()
        french_count = sum(1 for word in words[:100] if any(fw in word for fw in french_words))

        # Au moins 20% de mots français parmi les 100 premiers
        return french_count >= 20

    def scrape_context(self, context: str, sources: list[str]) -> None:
        """Scrape toutes les sources d'un contexte français"""
        logger.info(f"🚀 Début du scraping français pour le contexte '{context}'")

        all_articles = []

        for source in sources:
            articles = self.scrape_source(context, source)
            all_articles.extend(articles)

        # Sauvegarder les résultats
        context_dir = f"{OUTPUT_DIR}/{context}"
        os.makedirs(context_dir, exist_ok=True)

        output_file = f"{context_dir}/articles.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for article in all_articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        logger.info(f"💾 {len(all_articles)} articles français sauvegardés dans {output_file}")
        logger.info(
            f"✨ Scraping français terminé pour '{context}': {len(all_articles)} articles au total"
        )


def main():
    """Fonction principale pour le scraping français"""
    scraper = FrenchArticleScraper(max_articles_per_source=50, delay_range=(2, 4))

    for context, sources in FRENCH_SOURCES.items():
        try:
            scraper.scrape_context(context, sources)
            # Pause entre les contextes
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé par l'utilisateur")
            break
        except Exception as e:
            logger.error(f"❌ Erreur lors du scraping du contexte français '{context}': {e}")
            continue

    logger.info("🎉 Scraping français terminé pour tous les contextes")


if __name__ == "__main__":
    main()
