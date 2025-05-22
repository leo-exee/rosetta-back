# improved_text_scraper.py
import os
import random
import shutil
import time

import pandas as pd
import requests
import textstat
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def smart_throttle(min_delay=0.5, max_delay=1.5):
    time.sleep(random.uniform(min_delay, max_delay))


def log(msg: str, debug: bool = True):
    if debug:
        print(f"[LOG] {msg}")


ARTICLE_PATTERNS = {
    "bbc.com": {"selector": "a[data-testid='internal-link']", "filter": "/news/"},
    "cnn.com": {"selector": "a.container__link"},
    "theguardian.com": {"selector": "a[data-link-name='article']"},
}


class TextScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def _find_chrome_binary(self) -> str:
        paths = [
            shutil.which("google-chrome"),
            shutil.which("chromium-browser"),
            shutil.which("chromium"),
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
            "/usr/bin/google-chrome-stable",
            "/opt/google/chrome/chrome",
        ]
        return next((path for path in paths if path and os.path.exists(path)), None)

    def _setup_chrome_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        binary = self._find_chrome_binary()
        if binary:
            chrome_options.binary_location = binary
            log(f"Using Chrome binary at: {binary}")
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def scrape_news_articles(
        self, urls: list[str], max_articles: int = 10
    ) -> list[dict]:
        articles = []
        for url in urls:
            try:
                log(f"Scraping main page: {url}")
                soup = BeautifulSoup(
                    self.session.get(url, timeout=10).content, "html.parser"
                )
                domain = next((d for d in ARTICLE_PATTERNS if d in url), None)
                links = []
                if domain:
                    config = ARTICLE_PATTERNS[domain]
                    raw_links = soup.select(config["selector"])
                    links = [
                        l.get("href")
                        for l in raw_links
                        if l.get("href") and config.get("filter", "") in l.get("href")
                    ]
                else:
                    links = [
                        a.get("href")
                        for a in soup.find_all("a", href=True)
                        if "article" in a.get("href", "").lower()
                    ]
                full_links = self._normalize_urls(links, url)[:max_articles]
                log(f"Found {len(full_links)} article links")
                for article_url in full_links:
                    articles.extend(self._scrape_article(article_url))
                    smart_throttle()
            except Exception as e:
                log(f"Failed to scrape {url}: {e}")
        return articles

    def _normalize_urls(self, links: list[str], base_url: str) -> list[str]:
        base = f"https://{base_url.split('/')[2]}"
        result = set()
        for link in links:
            if link.startswith("/"):
                result.add(base + link)
            elif link.startswith("http"):
                result.add(link)
        return list(result)

    def _scrape_article(self, url: str) -> list[dict]:
        try:
            soup = BeautifulSoup(
                self.session.get(url, timeout=10).content, "html.parser"
            )
            title = next(
                (
                    soup.select_one(sel)
                    for sel in ["h1", ".headline", ".entry-title"]
                    if soup.select_one(sel)
                ),
                None,
            )
            if not title:
                return []
            paragraphs = soup.find_all("p")
            content = " ".join(
                p.get_text().strip()
                for p in paragraphs
                if len(p.get_text().strip()) > 30
            )
            if len(content.split()) > 50:
                return [
                    {
                        "title": title.get_text(strip=True),
                        "content": content,
                        "url": url,
                        "word_count": len(content.split()),
                    }
                ]
        except Exception as e:
            log(f"Error scraping article {url}: {e}")
        return []

    def scrape_educational_content(self) -> list[dict]:
        content = []
        content.extend(self._scrape_bbc_learning_requests())
        for url in [
            "https://www.englishclub.com",
            "https://www.perfect-english-grammar.com",
        ]:
            content.extend(self._scrape_generic_educational(url))
        return content

    def _scrape_bbc_learning_requests(self) -> list[dict]:
        try:
            soup = BeautifulSoup(
                self.session.get(
                    "https://www.bbc.co.uk/learningenglish", timeout=10
                ).content,
                "html.parser",
            )
            blocks = soup.find_all("div", class_=lambda x: x and "media" in x.lower())
            items = []
            for block in blocks[:10]:
                title = block.find(["h1", "h2", "h3"])
                desc = block.find("p")
                if title and desc:
                    items.append(
                        {
                            "title": title.get_text(strip=True),
                            "content": desc.get_text(strip=True),
                            "source": "BBC Learning",
                            "difficulty": "intermediate",
                        }
                    )
            return items
        except Exception as e:
            log(f"Error scraping BBC Learning: {e}")
            return []

    def _scrape_generic_educational(self, url: str) -> list[dict]:
        try:
            soup = BeautifulSoup(
                self.session.get(url, timeout=10).content, "html.parser"
            )
            paragraphs = [
                p.get_text().strip()
                for p in soup.find_all("p")
                if len(p.get_text().strip()) > 50
            ][:5]
            if paragraphs:
                return [
                    {
                        "title": f"Educational from {url}",
                        "content": " ".join(paragraphs),
                        "source": url,
                        "difficulty": "intermediate",
                    }
                ]
        except Exception as e:
            log(f"Error scraping {url}: {e}")
        return []

    def scrape_vocabulary_lists(self) -> list[dict]:
        try:
            soup = BeautifulSoup(
                self.session.get(
                    "https://www.merriam-webster.com/word-of-the-day", timeout=10
                ).content,
                "html.parser",
            )
            word = soup.find(["h1", "h2"], class_=lambda x: x and "word" in x.lower())
            definition = soup.find(
                ["p", "div"], class_=lambda x: x and "definition" in x.lower()
            )
            if word and definition:
                return [
                    {
                        "word": word.get_text(strip=True),
                        "definition": definition.get_text(strip=True),
                        "source": "Merriam-Webster",
                    }
                ]
        except Exception as e:
            log(f"Error scraping vocab: {e}")
        return []


class DatasetBuilder:
    def __init__(self):
        self.scraper = TextScraper()

    def _estimate_difficulty(self, text: str) -> str:
        try:
            score = textstat.flesch_reading_ease(text)
            return (
                "beginner"
                if score > 70
                else "intermediate" if score > 50 else "advanced"
            )
        except Exception as e:
            log(f"Error estimating difficulty: {e}")
            return "unknown"

    def build_complete_dataset(self) -> pd.DataFrame:
        news_urls = [
            "https://www.bbc.com/news",
            "https://edition.cnn.com",
            "https://www.theguardian.com/international",
        ]
        articles = self.scraper.scrape_news_articles(news_urls, max_articles=30)
        educational = self.scraper.scrape_educational_content()
        vocabulary = self.scraper.scrape_vocabulary_lists()

        data = []
        for a in articles:
            data.append(
                {
                    "text": a["content"],
                    "title": a["title"],
                    "type": "article",
                    "difficulty": self._estimate_difficulty(a["content"]),
                    "word_count": a["word_count"],
                }
            )
        for e in educational:
            data.append(
                {
                    "text": e["content"],
                    "title": e["title"],
                    "type": "educational",
                    "difficulty": e["difficulty"],
                    "word_count": len(e["content"].split()),
                }
            )
        for v in vocabulary:
            text = f"{v['word']}: {v['definition']}"
            data.append(
                {
                    "text": text,
                    "title": v["word"],
                    "type": "vocabulary",
                    "difficulty": "beginner",
                    "word_count": len(v["definition"].split()),
                }
            )

        df = pd.DataFrame(data)
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/scraped_data.csv", index=False)
        log("Dataset saved to data/raw/scraped_data.csv")
        return df


async def scrape_data_service():
    """Main service function to build the dataset."""
    builder = DatasetBuilder()
    builder.build_complete_dataset()
    print("✅ Scraping completed and dataset built.")
