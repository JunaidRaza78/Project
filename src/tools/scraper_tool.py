"""
Web Scraper Tool

Extracts content from web pages for deeper analysis with retry logic.
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class ScrapedContent:
    """Content extracted from a web page."""
    url: str
    title: str
    text: str
    meta_description: str = ""
    success: bool = True
    error: Optional[str] = None


class WebScraperTool:
    """
    Web content extractor with retry on transient errors.

    Fetches and parses web pages to extract relevant text content.
    """

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    BLOCKED_DOMAINS = {
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "linkedin.com",
    }

    # HTTP status codes that are retryable
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = None
        self._ensure_client()

    def _ensure_client(self):
        """Ensure the HTTP client is initialized and not closed."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.USER_AGENT},
                follow_redirects=True,
            )

    def _is_blocked_domain(self, url: str) -> bool:
        """Check if domain commonly blocks scraping."""
        try:
            domain = urlparse(url).netloc.lower()
            return any(blocked in domain for blocked in self.BLOCKED_DOMAINS)
        except Exception:
            return False

    async def scrape(self, url: str, max_length: int = 5000) -> ScrapedContent:
        """
        Scrape content from a URL with retry on transient errors.

        Args:
            url: URL to scrape
            max_length: Maximum text length to return

        Returns:
            ScrapedContent with extracted text
        """
        if self._is_blocked_domain(url):
            return ScrapedContent(
                url=url, title="", text="",
                success=False, error="Domain blocks automated access",
            )

        max_retries = 3
        last_error = ""

        for attempt in range(max_retries):
            try:
                self._ensure_client()

                response = await self.client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "lxml")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                # Get title
                title = ""
                if soup.title:
                    title = soup.title.get_text(strip=True)

                # Get meta description
                meta_desc = ""
                meta_tag = soup.find("meta", attrs={"name": "description"})
                if meta_tag:
                    meta_desc = meta_tag.get("content", "")

                # Get main content
                main_content = soup.find("article") or soup.find("main") or soup.find("body")

                if main_content:
                    text = main_content.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)

                # Clean up whitespace
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                text = "\n".join(lines)

                # Truncate if too long
                if len(text) > max_length:
                    text = text[:max_length] + "..."

                return ScrapedContent(
                    url=url, title=title, text=text,
                    meta_description=meta_desc, success=True,
                )

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {str(e)}"
                if e.response.status_code in self.RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break
            except httpx.HTTPError as e:
                last_error = f"HTTP error: {str(e)}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                break
            except Exception as e:
                last_error = f"Scraping error: {str(e)}"
                break

        return ScrapedContent(
            url=url, title="", text="",
            success=False, error=last_error,
        )

    async def scrape_multiple(
        self,
        urls: list[str],
        max_length: int = 3000,
    ) -> list[ScrapedContent]:
        """Scrape multiple URLs in parallel."""
        tasks = [self.scrape(url, max_length) for url in urls]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Close the HTTP client."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
