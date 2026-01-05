"""
Query Analyzer (from original Asta repo)

REFERENCE:
MOST OF THE FOLLOWING CODE IS A COPY FROM THE ASTA REPO:
https://github.com/allenai/asta-paper-finder, SEE LICENSE IN THAT REPO.

Note that this the query analyzer functionality is outside the scope of broad semantic search;
however, it is nessecary for the broad semantic search pipeline to function.
Thus, the original code has been copied here from the Asta repo.

Parses user query into structured form: query type, content, metadata filters, etc.
"""

from typing import Literal, Optional
import json

from pydantic import BaseModel, Field

# --- Types that would come from mabool/ai2i, defined locally ---

class Criterion(BaseModel):
    """Single relevance criterion with weight."""
    description: str
    weight: float = Field(ge=0, le=1)

class RelevanceCriteria(BaseModel):
    """Weighted sub-criteria for judging paper relevance."""
    criteria: list[Criterion] = Field(default_factory=list)

class DomainsIdentified(BaseModel):
    """Academic domains/fields for the query."""
    domains: list[str] = Field(default_factory=list)

class ExtractedYearlyTimeRange(BaseModel):
    """Year range filter."""
    start_year: Optional[int] = None
    end_year: Optional[int] = None


# --- Main types ---

class QueryType(BaseModel):
    type: Literal[
        "BROAD_SEMANTIC", 
        "SPECIFIC_BY_TITLE", 
        "SPECIFIC_BY_NAME", 
        "PURE_METADATA", 
        "CITING_PAPERS"
    ]

class AnalyzedQuery(BaseModel):
    """
    Structured output of the analyzer.
    Maps to what the pipeline needs to run a search.
    """
    query_type: QueryType
    content_query: str = Field(description="Semantic topic, stripped of metadata filters")
    relevance_criteria: RelevanceCriteria
    
    # Metadata filters
    time_range: Optional[ExtractedYearlyTimeRange] = None
    venues: Optional[list[str]] = None
    authors: Optional[list[str]] = None
    domains: DomainsIdentified = Field(default_factory=DomainsIdentified)
    
    # For specific paper lookups
    extracted_name: Optional[str] = Field(None, description="Name of specific paper if applicable")
    suitable_for_by_citing: Optional[bool] = None
    
    # Sorting preferences
    recent_first: bool = False
    recent_last: bool = False
    central_first: bool = False
    central_last: bool = False


class QueryAnalyzer:
    """
    Converts raw user query text into structured AnalyzedQuery.
    First step in the pipeline.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def analyze(self, user_query: str) -> AnalyzedQuery:
        prompt = f"""
You are an expert research assistant. Analyze the user's search query.

User Query: "{user_query}"

Extract the following:

1. **Query Type**:
   - SPECIFIC_BY_TITLE: Asks for a specific paper by exact title.
   - SPECIFIC_BY_NAME: Asks for a paper by fuzzy name (e.g., "the llama 3 paper").
   - PURE_METADATA: No semantic topic, only filters (e.g., "papers from NeurIPS 2024").
   - CITING_PAPERS: Asks for papers citing a specific paper.
   - BROAD_SEMANTIC: Default. Asks about a topic.

2. **Content Query**: 
   Core semantic topic. Remove "papers about", "find me", and metadata filters.
   Example: "papers by LeCun on energy based models" -> "energy based models"

3. **Metadata**:
   Authors, Venues (conferences/journals), Time Range (years).
   Domains: Academic field (e.g., Computer Science, Biology).

4. **Sorting Intent**:
   - recent_first: "new", "latest", "recent"
   - central_first: "seminal", "important", "highly cited", "classic"

5. **Relevance Criteria**:
   Generate 3-5 specific sub-criteria to judge papers by.
   Assign weights summing to 1.0.

Output as JSON matching this schema:
{{
  "query_type": {{"type": "BROAD_SEMANTIC"}},
  "content_query": "...",
  "relevance_criteria": {{
    "criteria": [
      {{"description": "...", "weight": 0.4}},
      ...
    ]
  }},
  "time_range": {{"start_year": 2020, "end_year": 2024}} or null,
  "venues": ["NeurIPS", "ICML"] or null,
  "authors": ["Yann LeCun"] or null,
  "domains": {{"domains": ["Computer Science"]}},
  "recent_first": false,
  "central_first": false
}}
"""

        response = await self.llm_client.generate(
            prompt=prompt,
            response_format={"type": "json_object"},
            label="analyzing query",
        )
        
        # Strip markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            # Remove ```json and trailing ```
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        data = json.loads(content)
        return AnalyzedQuery(**data)
