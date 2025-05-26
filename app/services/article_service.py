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

# Sources améliorées et pertinentes
SOURCES = {
    "it": [
        "https://techcrunch.com/",
        "https://www.wired.com/category/security/",
        "https://arstechnica.com/",
        "https://www.theverge.com/tech",
        "https://venturebeat.com/",
    ],
    "work": [
        "https://hbr.org/",
        "https://www.entrepreneur.com/",
        "https://www.forbes.com/business/",
        "https://www.inc.com/",
        "https://www.fastcompany.com/",
    ],
    "travel": [
        "https://www.lonelyplanet.com/articles",
        "https://www.nomadicmatt.com/travel-blog/",
        "https://www.travelandleisure.com/",
        "https://www.cntraveler.com/",
        "https://www.afar.com/",
    ],
}

OUTPUT_DIR = "data/articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


class ArticleScraper:
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
        """Extrait les liens d'articles de manière plus intelligente"""
        links = set()
        domain = urlparse(base_url).netloc

        # Sélecteurs CSS spécifiques par site pour de meilleurs résultats
        selectors = [
            'a[href*="/article"]',
            'a[href*="/story"]',
            'a[href*="/post"]',
            'a[href*="/blog"]',
            "article a",
            ".entry-title a",
            ".post-title a",
            "h2 a",
            "h3 a",
        ]

        for selector in selectors:
            for link in soup.select(selector):
                href = link.get("href", "")
                if href:
                    full_url = urljoin(base_url, href)
                    # Vérifier que c'est bien un article du même domaine
                    if domain in full_url and self.is_article_url(full_url):
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
                if domain in full_url and self.is_article_url(full_url):
                    links.add(full_url)
                    if len(links) >= self.max_articles_per_source:
                        break

        return list(links)

    def is_article_url(self, url: str) -> bool:
        """Détermine si une URL semble être un article"""
        url_lower = url.lower()

        # URLs à éviter
        avoid_patterns = [
            "/tag/",
            "/category/",
            "/author/",
            "/page/",
            "/search/",
            "/login",
            "/register",
            "/contact",
            "/about",
            "/privacy",
            ".pdf",
            ".jpg",
            ".png",
            ".gif",
            ".mp4",
            ".zip",
            "#",
            "javascript:",
            "mailto:",
            "tel:",
        ]

        if any(pattern in url_lower for pattern in avoid_patterns):
            return False

        # URLs probablement intéressantes
        good_patterns = [
            "/article",
            "/story",
            "/post",
            "/blog",
            "/news",
            "/2024/",
            "/2025/",
            "/how-",
            "/what-",
            "/why-",
        ]

        return (
            any(pattern in url_lower for pattern in good_patterns)
            or len(url.split("/")) >= 4
        )

    def extract_article_content(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extrait le titre et le contenu d'un article"""
        # Extraction du titre
        title = ""
        title_selectors = [
            "h1",
            ".entry-title",
            ".post-title",
            ".article-title",
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
        for element in soup(
            ["script", "style", "nav", "header", "footer", "aside", "advertisement"]
        ):
            element.decompose()

        # Chercher le contenu principal
        content_selectors = [
            ".entry-content",
            ".post-content",
            ".article-content",
            ".content",
            "article",
            ".story-body",
            "main",
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
        """Scrape une source spécifique"""
        logger.info(f"🔍 Scraping {source_url} pour le contexte '{context}'")

        # Récupérer la page principale
        html = self.fetch_html(source_url)
        if not html:
            logger.warning(f"❌ Impossible de récupérer {source_url}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        article_links = self.extract_article_links(source_url, soup)

        logger.info(f"📄 {len(article_links)} liens d'articles trouvés")

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

                # Filtrer les articles trop courts ou sans contenu
                if len(article_data["content"].split()) < 100:
                    logger.debug(f"⚠️ Article trop court ignoré: {link}")
                    continue

                articles.append(
                    {
                        "context": context,
                        "source": urlparse(source_url).netloc,
                        "url": link,
                        "title": article_data["title"],
                        "content": article_data["content"],
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                logger.info(
                    f"✅ Article {i+1}/{len(article_links)} traité: {article_data['title'][:50]}..."
                )

            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {link}: {e}")
                continue

        logger.info(f"🎯 {len(articles)} articles valides récupérés de {source_url}")
        return articles

    def scrape_context(self, context: str, sources: list[str]) -> None:
        """Scrape tous les sources d'un contexte"""
        logger.info(f"🚀 Début du scraping pour le contexte '{context}'")

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

        logger.info(f"💾 {len(all_articles)} articles sauvegardés dans {output_file}")
        logger.info(
            f"✨ Scraping terminé pour '{context}': {len(all_articles)} articles au total"
        )


def main():
    """Fonction principale"""
    scraper = ArticleScraper(max_articles_per_source=50, delay_range=(2, 4))

    for context, sources in SOURCES.items():
        try:
            scraper.scrape_context(context, sources)
            # Pause entre les contextes
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé par l'utilisateur")
            break
        except Exception as e:
            logger.error(f"❌ Erreur lors du scraping du contexte '{context}': {e}")
            continue

    logger.info("🎉 Scraping terminé pour tous les contextes")


if __name__ == "__main__":
    main()
