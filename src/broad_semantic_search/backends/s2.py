"""
Semantic Scholar API client.

Uses two search modes:
- Semantic Search: /paper/search - finds papers by meaning
- Keyword Search: /paper/search/bulk - boolean keyword matching
"""

import logging
import os
from typing import Optional, List, Dict

import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class S2Paper:
    """A paper from Semantic Scholar."""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citation_count: int
    authors: List[str]
    url: Optional[str]
    
    # Which search found this paper
    source: str  # "semantic" or "keyword"


class S2Client:
    """Client for Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Fields we want for each paper
    FIELDS = "paperId,title,abstract,year,citationCount,authors,url"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("S2_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    async def semantic_search(self, query: str, limit: int = 20) -> List[S2Paper]:
        """
        Semantic (dense) search - finds papers by meaning.
        Uses /paper/search endpoint which ranks by relevance.
        """
        logger.info(f"S2 semantic search: {query!r}")
        url = f"{self.BASE_URL}/paper/search" # THIS IS NOT ENTIRELY SEMANTIC DENSE SEARCH, NEED IMPROVEMENT
        params = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "fields": self.FIELDS,
        }
        
        resp = await self.client.get(url, params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        
        papers = []
        for item in data.get("data", []):
            papers.append(self._parse_paper(item, source="semantic"))
        return papers
    
    async def keyword_search(self, query: str, limit: int = 20) -> List[S2Paper]:
        """
        Keyword (bulk) search - boolean keyword matching.
        Uses /paper/search/bulk which supports operators: + | - "phrase" 
        """
        logger.info(f"S2 keyword search: {query!r}")
        url = f"{self.BASE_URL}/paper/search/bulk"
        params = {
            "query": query,
            "fields": self.FIELDS,
        }
        
        resp = await self.client.get(url, params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        
        # Bulk search can return many results, we take first N
        papers = []
        for item in data.get("data", [])[:limit]:
            papers.append(self._parse_paper(item, source="keyword"))
        return papers
    
    def _parse_paper(self, item: dict, source: str) -> S2Paper:
        """Convert API response to S2Paper."""
        authors = []
        for author in item.get("authors", []):
            if author and author.get("name"):
                authors.append(author["name"])
        
        return S2Paper(
            paper_id=item.get("paperId", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract"),
            year=item.get("year"),
            citation_count=item.get("citationCount", 0) or 0,
            authors=authors,
            url=item.get("url"),
            source=source,
        )
    
    async def close(self):
        await self.client.aclose()
