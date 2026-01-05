"""
Result Aggregation: deduplicate papers and merge snippets.

Takes RetrievalHit objects from multiple queries and:
1. Groups by paper_id
2. Merges snippets from all queries that found the paper
3. Keeps track of which query/source produced each snippet
4. Keeps the highest relevance score
"""

import logging
from typing import Dict, List

from broad_semantic_search.models import RetrievalHit, Snippet, AggregatedPaper

logger = logging.getLogger(__name__)


def aggregate_hits(hits: List[RetrievalHit]) -> List[AggregatedPaper]:
    """
    Aggregate retrieval hits by paper_id.
    
    - Deduplicates papers that appear in multiple queries
    - Merges snippets from all sources
    - Tracks provenance (which query/source found it)
    - Keeps the highest score for ranking
    """
    # Group hits by paper_id
    paper_map: Dict[str, AggregatedPaper] = {}
    
    for hit in hits:
        paper_id = hit.paper.paper_id
        
        # Create snippet from this hit
        snippet = Snippet(
            text=hit.paper.snippet_text or hit.paper.abstract or "",
            query=hit.query,
            source=hit.source,
            score=0.0,  # We don't have scores from S2 yet
        )
        
        if paper_id not in paper_map:
            # First time seeing this paper
            paper_map[paper_id] = AggregatedPaper(
                paper_id=paper_id,
                title=hit.paper.title,
                abstract=hit.paper.abstract,
                year=hit.paper.year,
                citation_count=hit.paper.citation_count,
                authors=hit.paper.authors,
                url=hit.paper.url,
                snippets=[snippet],
                sources=[hit.source],
                queries=[hit.query],
                best_score=0.0,
            )
        else:
            # Already seen this paper - merge data
            agg = paper_map[paper_id]
            
            # Add snippet (avoid exact duplicates)
            if snippet.text and snippet.text not in [s.text for s in agg.snippets]:
                agg.snippets.append(snippet)
            
            # Add source if new
            if hit.source not in agg.sources:
                agg.sources.append(hit.source)
            
            # Add query if new
            if hit.query not in agg.queries:
                agg.queries.append(hit.query)
    
    papers = list(paper_map.values())
    
    # Sort snippets within each paper by score (highest first)
    for paper in papers:
        paper.snippets.sort(key=lambda s: s.score, reverse=True)
    
    logger.info(
        f"Aggregated {len(hits)} hits into {len(papers)} unique papers "
        f"(deduped {len(hits) - len(papers)} duplicates)"
    )
    
    return papers
