"""
Serper Search Tool

Real-time Google search using Serper API.
"""

import httpx
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..state import SearchResult


@dataclass
class SerperSearchConfig:
    """Configuration for Serper API."""
    api_key: str
    base_url: str = "https://google.serper.dev/search"
    num_results: int = 10
    country: str = "us"
    language: str = "en"


class SerperSearchTool:
    """
    Serper API integration for Google search.
    
    Provides real-time web search results for the research agent.
    """
    
    def __init__(self, api_key: str, num_results: int = 10):
        self.config = SerperSearchConfig(
            api_key=api_key,
            num_results=num_results,
        )
        self.client: httpx.AsyncClient | None = None
        self._ensure_client()
    
    def _ensure_client(self):
        """Ensure the HTTP client is initialized and not closed."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(
        self,
        query: str,
        search_type: str = "search",  # search, news, images
        num_results: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Perform a Google search via Serper API.
        
        Args:
            query: Search query string
            search_type: Type of search (search, news, images)
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Ensure client is available
            self._ensure_client()
            
            headers = {
                "X-API-KEY": self.config.api_key,
                "Content-Type": "application/json",
            }
            
            payload = {
                "q": query,
                "gl": self.config.country,
                "hl": self.config.language,
                "num": num_results or self.config.num_results,
            }
            
            # Choose endpoint based on search type
            if search_type == "news":
                url = "https://google.serper.dev/news"
            elif search_type == "images":
                url = "https://google.serper.dev/images"
            else:
                url = self.config.base_url
            
            response = await self.client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Parse organic results
            organic = data.get("organic", [])
            for item in organic:
                result = SearchResult(
                    query=query,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                    timestamp=datetime.now(),
                    relevance_score=item.get("position", 10) / 10,  # Invert position for score
                )
                results.append(result)
            
            # Also include knowledge graph if available
            kg = data.get("knowledgeGraph", {})
            if kg:
                kg_result = SearchResult(
                    query=query,
                    title=kg.get("title", "Knowledge Graph"),
                    snippet=kg.get("description", ""),
                    url=kg.get("website", ""),
                    timestamp=datetime.now(),
                    relevance_score=1.0,  # High relevance for KG
                )
                results.insert(0, kg_result)
            
            return results
            
        except httpx.HTTPError as e:
            # Return empty results on error, let caller handle
            return []
    
    async def search_news(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search news articles."""
        return await self.search(query, search_type="news", num_results=num_results)
    
    async def multi_search(self, queries: list[str]) -> dict[str, list[SearchResult]]:
        """
        Perform multiple searches in parallel.
        
        Args:
            queries: List of search queries
            
        Returns:
            Dictionary mapping queries to their results
        """
        import asyncio
        
        tasks = [self.search(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        return dict(zip(queries, results))
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
