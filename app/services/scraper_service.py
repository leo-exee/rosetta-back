import os
import platform
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class TextScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def scrape_news_articles(self, urls: list[str]) -> list[dict]:
        """Scrapes news articles (title + text content) from given URLs."""
        articles = []

        for url in urls:
            try:
                response = self.session.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                # This part must be customized per site structure
                title = soup.find("h1")
                content = soup.find_all("p")

                if title and content:
                    article_text = " ".join([p.get_text().strip() for p in content])
                    articles.append(
                        {
                            "title": title.get_text().strip(),
                            "content": article_text,
                            "url": url,
                            "word_count": len(article_text.split()),
                        }
                    )

                time.sleep(random.uniform(1, 3))  # Respect site's rate limiting

            except Exception as e:
                print(f"Error scraping {url}: {e}")

        return articles

    def scrape_educational_content(self) -> list[dict]:
        """Scrapes English learning content from educational websites."""
        sources = [
            "https://www.bbc.co.uk/learningenglish",
            "https://www.englishclub.com",
            "https://www.perfect-english-grammar.com",
        ]

        educational_content = []

        for source in sources:
            driver = None
            try:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")

                # Chrome binary path for WSL
                if "microsoft" in platform.uname().release.lower():
                    chrome_options.binary_location = "/usr/bin/google-chrome"

                # Install and initialize ChromeDriver
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)

                # Use appropriate scraping method depending on the site
                if "bbc" in source:
                    content = self._scrape_bbc_learning(source)
                elif "englishclub" in source:
                    content = self._scrape_english_club(source)
                else:
                    content = self._scrape_generic(source)

                educational_content.extend(content)

            except Exception as e:
                print(f"Error scraping {source}: {e}")

            finally:
                if driver:
                    driver.quit()  # Ensure browser closes after use

        return educational_content

    def _scrape_bbc_learning(self, url: str) -> list[dict]:
        """Specific scraper for BBC Learning English."""
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        articles = soup.find_all("article", class_="media")
        content = []

        for article in articles:
            title_elem = article.find("h3")
            desc_elem = article.find("p")

            if title_elem and desc_elem:
                content.append(
                    {
                        "title": title_elem.get_text().strip(),
                        "content": desc_elem.get_text().strip(),
                        "source": "BBC Learning English",
                        "difficulty": "intermediate",
                    }
                )

        return content

    def _scrape_english_club(self, url: str) -> list[dict]:
        """Placeholder for scraping EnglishClub (should be customized)."""
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text().strip() for p in paragraphs[:5])

        return [
            {
                "title": "EnglishClub Content",
                "content": text,
                "source": "EnglishClub",
                "difficulty": "intermediate",
            }
        ]

    def _scrape_generic(self, url: str) -> list[dict]:
        """Generic fallback scraper for educational content."""
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text().strip() for p in paragraphs[:5])

        return [
            {
                "title": "Generic Content",
                "content": text,
                "source": url,
                "difficulty": "intermediate",
            }
        ]

    def scrape_vocabulary_lists(self) -> list[dict]:
        """Scrapes vocabulary lists with definitions from specific websites."""
        vocab_sources = [
            "https://www.vocabulary.com/lists/",
            "https://www.merriam-webster.com/word-of-the-day",
        ]

        vocabulary = []

        for source in vocab_sources:
            try:
                response = self.session.get(source)
                soup = BeautifulSoup(response.content, "html.parser")

                # Basic structure detection – may require adjustments
                word_elements = soup.find_all("div", class_="word")

                for word_elem in word_elements:
                    word = word_elem.find("h3")
                    definition = word_elem.find("p", class_="definition")

                    if word and definition:
                        vocabulary.append(
                            {
                                "word": word.get_text().strip(),
                                "definition": definition.get_text().strip(),
                                "source": source,
                            }
                        )

            except Exception as e:
                print(f"Error scraping vocabulary from {source}: {e}")

        return vocabulary


class DatasetBuilder:
    def __init__(self):
        self.scraper = TextScraper()

    def build_complete_dataset(self) -> pd.DataFrame:
        """Builds a complete dataset from articles, educational content, and vocabulary."""
        news_urls = [
            "https://www.bbc.com/news",
            "https://edition.cnn.com",
            "https://www.theguardian.com/international",
        ]

        print("Scraping news articles...")
        articles = self.scraper.scrape_news_articles(news_urls[:5])  # Limit for testing

        print("Scraping educational content...")
        educational = self.scraper.scrape_educational_content()

        print("Scraping vocabulary lists...")
        vocabulary = self.scraper.scrape_vocabulary_lists()

        all_data = []

        # Process news articles
        for article in articles:
            all_data.append(
                {
                    "text": article["content"],
                    "title": article["title"],
                    "type": "article",
                    "difficulty": self._estimate_difficulty(article["content"]),
                    "word_count": article["word_count"],
                }
            )

        # Process educational content
        for content in educational:
            all_data.append(
                {
                    "text": content["content"],
                    "title": content["title"],
                    "type": "educational",
                    "difficulty": content.get("difficulty", "intermediate"),
                    "word_count": len(content["content"].split()),
                }
            )

        # Process vocabulary items
        for vocab in vocabulary:
            all_data.append(
                {
                    "text": f"{vocab['word']}: {vocab['definition']}",
                    "title": vocab["word"],
                    "type": "vocabulary",
                    "difficulty": "beginner",
                    "word_count": len(vocab["definition"].split()),
                }
            )

        df = pd.DataFrame(all_data)

        # Ensure the directory exists before saving
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/scraped_data.csv", index=False)

        return df

    def _estimate_difficulty(self, text: str) -> str:
        """Estimates difficulty of text based on average word length."""
        words = text.split()
        if not words:
            return "unknown"

        avg_word_length = sum(len(word) for word in words) / len(words)

        if avg_word_length < 4.5:
            return "beginner"
        elif avg_word_length < 6:
            return "intermediate"
        else:
            return "advanced"


async def scrape_data_service():
    dataset_builder = DatasetBuilder()
    dataset_builder.build_complete_dataset()
