"""
Broad Search Agent

Runs the pipeline: analyze -> reformulate -> retrieve -> dedupe -> judge -> rank (aka orchestrator)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from broad_semantic_search.third_party.analyze import QueryAnalyzer, AnalyzedQuery
from broad_semantic_search.reformulate import get_reformulated_queries
from broad_semantic_search.retrieve import run_retrieval
from broad_semantic_search.aggregate import aggregate_hits
from broad_semantic_search.judge import judge_papers
from broad_semantic_search.backends.s2 import S2Client

logger = logging.getLogger(__name__)


class BroadSearchAgent:
    """Wires together all the pipeline stages."""
    
    def __init__(self, llm_client=None, s2_client: Optional[S2Client] = None, max_results: int = 20):
        self.llm_client = llm_client
        self.s2_client = s2_client
        self.max_results = max_results
        self.analyzer = QueryAnalyzer(llm_client) if llm_client else None
        
        # Configure logging - suppress noisy httpx logs
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    async def run(self, query: str) -> dict[str, Any]:
        """Run the full pipeline on a user query."""
        logger.info(f"Starting broad search for: {query!r}")
        
        # --- Step 0: Analyze query ---
        analyzed: Optional[AnalyzedQuery] = None
        if self.analyzer:
            analyzed = await self.analyzer.analyze(query)
            logger.info(f"Query type: {analyzed.query_type.type}")
            logger.info(f"Content query: {analyzed.content_query}")
            content_query = analyzed.content_query
        else:
            # No LLM client, use raw query
            content_query = query
        
        # --- Step 1: Reformulate ---
        if self.llm_client:
            reformulations = await get_reformulated_queries(content_query, self.llm_client, k=3)
        else:
            reformulations = [content_query]
        logger.info(f"Reformulations ({len(reformulations)}): {reformulations}")
        
        # --- Step 2: Retrieve ---
        hits = []
        if self.s2_client:
            hits = await run_retrieval(reformulations, self.s2_client, limit_per_source=10)
        logger.info(f"Retrieved {len(hits)} hits")
        
        # Log sample of what we got
        for hit in hits[:3]:
            logger.info(f"  - [{hit.source}] {hit.paper.title[:60]}...")
        
        # --- Step 3: Aggregate (dedupe + merge snippets) ---
        papers = aggregate_hits(hits)
        
        # Log sample of aggregated papers
        for paper in papers[:3]:
            logger.info(
                f"  - {paper.title[:50]}... "
                f"(sources: {paper.sources}, queries: {len(paper.queries)}, snippets: {len(paper.snippets)})"
            )
        
        # --- Step 4: Judge ---
        judged_papers = []
        if self.llm_client and papers:
            judged_papers = await judge_papers(
                papers=papers,
                query=content_query,
                analyzed=analyzed,
                llm_client=self.llm_client,
            )
        
        # Count by label
        label_counts = {}
        for p in judged_papers:
            label_counts[p.label] = label_counts.get(p.label, 0) + 1
        logger.info(f"Judgment labels: {label_counts}")
        
        # --- Step 5: Rank ---
        # Sort by final_score descending
        ranked_papers = sorted(judged_papers, key=lambda p: p.final_score, reverse=True)
        logger.info(f"Ranked {len(ranked_papers)} papers")
        
        # Log top 3 with scores
        for p in ranked_papers[:3]:
            logger.info(
                f"  - [{p.label}] {p.title[:50]}... "
                f"(final={p.final_score:.2f}, semantic={p.semantic_score:.2f}, cite={p.citation_score:.2f})"
            )
        
        return {
            "query": query,
            "analyzed": analyzed.model_dump() if analyzed else None,
            "papers": [self._paper_to_dict(p) for p in ranked_papers[:self.max_results]],
            "metadata": {
                "total_retrieved": len(hits),
                "unique_papers": len(papers),
                "highly_relevant": label_counts.get("highly_relevant", 0),
                "relevant": label_counts.get("relevant", 0),
                "somewhat_relevant": label_counts.get("somewhat_relevant", 0),
                "not_relevant": label_counts.get("not_relevant", 0),
                "search_strategy": "broad_semantic_search",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
    
    def _paper_to_dict(self, paper) -> dict:
        """Convert JudgedPaper to serializable dict."""
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "citation_count": paper.citation_count,
            "authors": paper.authors,
            "url": paper.url,
            "label": paper.label,
            "final_score": round(paper.final_score, 3),
            "semantic_score": round(paper.semantic_score, 3),
            "citation_score": round(paper.citation_score, 3),
            "recency_score": round(paper.recency_score, 3),
            "reasoning": paper.reasoning,
        }
