"""
Retrieval stage: fetch papers for each query using hybrid search.

For each query, we run:
1. Semantic (dense) search - finds papers by meaning
2. Keyword search - finds papers with exact keyword matches

Returns a list of RetrievalHit objects (one per paper per query).
"""

import asyncio
import logging

from broad_semantic_search.backends.s2 import S2Client, S2Paper
from broad_semantic_search.models import Paper, RetrievalHit

logger = logging.getLogger(__name__)


def s2_to_paper(s2paper: S2Paper) -> Paper:
    """Convert S2Paper to our Paper model."""
    return Paper(
        paper_id=s2paper.paper_id,
        title=s2paper.title,
        abstract=s2paper.abstract,
        year=s2paper.year,
        citation_count=s2paper.citation_count,
        authors=s2paper.authors,
        url=s2paper.url,
        sources=[s2paper.source],
        snippet_text=s2paper.abstract[:200] if s2paper.abstract else None,
    )


async def retrieve_for_query(
    query: str,
    s2_client: S2Client,
    limit_per_source: int = 10,
) -> list[RetrievalHit]:
    """
    Run hybrid retrieval for a single query.
    Returns hits from both semantic and keyword search.
    """
    hits = []
    
    # Run searches sequentially to avoid rate limiting
    try:
        semantic_results = await s2_client.semantic_search(query, limit=limit_per_source)
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        semantic_results = []
    
    # Delay between search types to avoid rate limiting
    await asyncio.sleep(2.0)
    
    try:
        keyword_results = await s2_client.keyword_search(query, limit=limit_per_source)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        keyword_results = []
    
    # Process semantic results
    for s2paper in semantic_results:
        paper = s2_to_paper(s2paper)
        paper.queries.append(query)
        hits.append(RetrievalHit(paper=paper, query=query, source="semantic"))
    if semantic_results:
        logger.info(f"Semantic search for {query!r}: {len(semantic_results)} papers")
    
    # Process keyword results
    for s2paper in keyword_results:
        paper = s2_to_paper(s2paper)
        paper.queries.append(query)
        hits.append(RetrievalHit(paper=paper, query=query, source="keyword"))
    if keyword_results:
        logger.info(f"Keyword search for {query!r}: {len(keyword_results)} papers")
    
    return hits


async def run_retrieval(
    queries: list[str],
    s2_client: S2Client,
    limit_per_source: int = 10,
    delay_seconds: float = 3.0,
) -> list[RetrievalHit]:
    """
    Run hybrid retrieval for all queries.
    Returns all hits (not yet deduplicated).
    
    Args:
        delay_seconds: Delay between queries to avoid rate limiting (default 3s for public API).
    """
    all_hits = []
    
    # Run queries sequentially with delay to avoid rate limiting
    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(delay_seconds)
            logger.info(f"Processing query {i+1}/{len(queries)}: {query!r}")
        hits = await retrieve_for_query(query, s2_client, limit_per_source)
        all_hits.extend(hits)
    
    logger.info(f"Total retrieval hits: {len(all_hits)}")
    return all_hits
