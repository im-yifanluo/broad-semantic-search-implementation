"""
LLM Relevance Judgment

Scores each paper using:
1. LLM semantic judgment (how well does it match the query?)
2. Citation score (normalized, log scale)
3. Recency score (boost recent papers if requested)

Outputs labeled papers: highly_relevant, relevant, somewhat_relevant, not_relevant
"""

import json
import logging
import math
from datetime import datetime
from typing import List, Optional, Dict

from broad_semantic_search.models import AggregatedPaper, JudgedPaper, Snippet
from broad_semantic_search.third_party.analyze import AnalyzedQuery, Criterion

logger = logging.getLogger(__name__)

# Score weights
DEFAULT_WEIGHTS = {
    "semantic": 0.6,
    "citation": 0.25,
    "recency": 0.15,
}

# Label thresholds
LABEL_THRESHOLDS = {
    "highly_relevant": 0.8,
    "relevant": 0.6,
    "somewhat_relevant": 0.3,
}


def compute_citation_score(citation_count: int, max_citations: int = 10000) -> float:
    """
    Normalize citation count to 0-1 using log scale.
    Papers with 0 citations get 0, papers with 10k+ get ~1.
    """
    if citation_count <= 0:
        return 0.0
    # Log scale: log(1) = 0, log(10000) ≈ 9.2
    log_count = math.log(citation_count + 1)
    log_max = math.log(max_citations + 1)
    return min(log_count / log_max, 1.0)


def compute_recency_score(year: Optional[int], recent_first: bool = False) -> float:
    """
    Score based on publication year.
    If recent_first=True, boost recent papers.
    Otherwise, return neutral 0.5.
    """
    if year is None:
        return 0.5  # neutral for unknown year
    
    if not recent_first:
        return 0.5  # no preference
    
    current_year = datetime.now().year
    age = current_year - year
    
    # Papers from last 2 years get high scores, older papers decay
    if age <= 0:
        return 1.0
    elif age <= 2:
        return 0.9
    elif age <= 5:
        return 0.7
    elif age <= 10:
        return 0.5
    else:
        return 0.3


def score_to_label(score: float) -> str:
    """Convert relevance score to label."""
    if score >= LABEL_THRESHOLDS["highly_relevant"]:
        return "highly_relevant"
    elif score >= LABEL_THRESHOLDS["relevant"]:
        return "relevant"
    elif score >= LABEL_THRESHOLDS["somewhat_relevant"]:
        return "somewhat_relevant"
    else:
        return "not_relevant"


async def judge_paper_batch(
    papers: List[AggregatedPaper],
    query: str,
    criteria: List[Criterion],
    llm_client,
) -> List[Dict]:
    """
    Use LLM to judge relevance of a batch of papers.
    Returns list of {paper_id, score, reasoning}.
    """
    if not papers:
        return []
    
    # Build criteria text
    criteria_text = "\n".join(
        f"- {c.description} (weight: {c.weight})" for c in criteria
    )
    
    # Build papers text (limit to first 10 for batch)
    papers_text = ""
    for i, paper in enumerate(papers[:10]):
        snippet = paper.snippets[0].text[:300] if paper.snippets else (paper.abstract[:300] if paper.abstract else "No abstract")
        papers_text += f"""
Paper {i+1}:
- ID: {paper.paper_id}
- Title: {paper.title}
- Year: {paper.year}
- Snippet: {snippet}...
"""

    prompt = f"""You are a research paper relevance judge.

Query: "{query}"

Relevance Criteria:
{criteria_text}

Papers to judge:
{papers_text}

For each paper, score its relevance from 0.0 to 1.0 based on how well it matches the query and criteria.

Respond with a JSON array:
[
  {{"paper_id": "...", "score": 0.85, "reasoning": "brief explanation"}},
  ...
]

Only output the JSON array, nothing else."""

    try:
        response = await llm_client.generate(
            prompt=prompt,
            response_format={"type": "json_object"},
            label=f"scoring {len(papers)} papers",
        )
        
        # Parse response - handle markdown wrapping
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        judgments = json.loads(text)
        
        # Handle if wrapped in object
        if isinstance(judgments, dict) and "papers" in judgments:
            judgments = judgments["papers"]
        elif isinstance(judgments, dict) and "judgments" in judgments:
            judgments = judgments["judgments"]
        
        return judgments
        
    except Exception as e:
        logger.error(f"LLM judgment failed: {e}")
        # Return neutral scores on failure
        return [{"paper_id": p.paper_id, "score": 0.5, "reasoning": "judgment failed"} for p in papers]


async def judge_papers(
    papers: List[AggregatedPaper],
    query: str,
    analyzed: Optional[AnalyzedQuery],
    llm_client,
    weights: Optional[Dict[str, float]] = None,
) -> List[JudgedPaper]:
    """
    Judge all papers and compute final scores.
    
    Returns papers with:
    - semantic_score (from LLM)
    - citation_score (log normalized)
    - recency_score (if recent_first)
    - final_score (weighted combination)
    - label (highly_relevant, relevant, etc.)
    """
    if not papers:
        return []
    
    weights = weights or DEFAULT_WEIGHTS
    criteria = analyzed.relevance_criteria.criteria if analyzed else []
    recent_first = analyzed.recent_first if analyzed else False
    
    # Get LLM judgments in batches
    all_judgments = []
    batch_size = 10
    total_batches = (len(papers) + batch_size - 1) // batch_size
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Judging batch {batch_num}/{total_batches} ({len(batch)} papers)")
        judgments = await judge_paper_batch(batch, query, criteria, llm_client)
        all_judgments.extend(judgments)
    
    # Map judgments by paper_id
    judgment_map = {j["paper_id"]: j for j in all_judgments}
    
    # Compute all scores and create JudgedPaper objects
    judged = []
    for paper in papers:
        j = judgment_map.get(paper.paper_id, {"score": 0.5, "reasoning": "not judged"})
        
        semantic_score = j.get("score", 0.5)
        citation_score = compute_citation_score(paper.citation_count)
        recency_score = compute_recency_score(paper.year, recent_first)
        
        # Weighted final score
        final_score = (
            weights["semantic"] * semantic_score +
            weights["citation"] * citation_score +
            weights["recency"] * recency_score
        )
        
        label = score_to_label(final_score)
        
        judged.append(JudgedPaper(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
            authors=paper.authors,
            url=paper.url,
            snippets=paper.snippets,
            label=label,
            relevance_score=semantic_score,
            reasoning=j.get("reasoning", ""),
            semantic_score=semantic_score,
            citation_score=citation_score,
            recency_score=recency_score,
            final_score=final_score,
        ))
    
    # Count labels
    label_counts = {}
    for p in judged:
        label_counts[p.label] = label_counts.get(p.label, 0) + 1
    
    logger.info(f"Judged {len(judged)} papers: {label_counts}")
    
    return judged
