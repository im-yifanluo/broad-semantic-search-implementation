"""
Data models for the pipeline.

The data structures used throughout the broad semantic search pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Snippet:
    """A text snippet from a paper, with provenance."""
    text: str
    query: str  # which query produced this snippet
    source: str  # "semantic" or "keyword"
    score: float = 0.0  # relevance score from retrieval


@dataclass
class Paper:
    """A paper in our pipeline, with retrieval metadata."""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citation_count: int
    authors: List[str]
    url: Optional[str]
    
    # How this paper was found
    sources: List[str] = field(default_factory=list)  # e.g. ["semantic", "keyword"]
    queries: List[str] = field(default_factory=list)  # which queries found it
    
    # Snippet text (the relevant excerpt that matched)
    snippet_text: Optional[str] = None


@dataclass
class AggregatedPaper:
    """A paper after aggregation - deduped with multiple snippets."""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citation_count: int
    authors: List[str]
    url: Optional[str]
    
    # Aggregated data
    snippets: List[Snippet] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # unique sources
    queries: List[str] = field(default_factory=list)  # unique queries that found it
    best_score: float = 0.0  # highest relevance score across all hits


@dataclass
class RetrievalHit:
    """A single retrieval result before merging."""
    paper: Paper
    query: str
    source: str  # "semantic" or "keyword"


@dataclass
class JudgedPaper:
    """A paper after LLM judgment with scores and label."""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citation_count: int
    authors: List[str]
    url: Optional[str]
    snippets: List[Snippet]
    
    # Judgment results
    label: str  # "highly_relevant", "relevant", "somewhat_relevant", "not_relevant"
    relevance_score: float  # 0-1, from LLM judgment
    reasoning: str  # LLM explanation
    
    # Component scores (all 0-1)
    semantic_score: float = 0.0
    citation_score: float = 0.0
    recency_score: float = 0.0
    
    # Final weighted score
    final_score: float = 0.0
