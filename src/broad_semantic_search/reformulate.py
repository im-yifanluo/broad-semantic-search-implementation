"""
Query reformulation: generate semantic variations of the user's query.

Ask the LLM to produce k alternative phrasings of the query that capture the same intent but use different wording.
Returns k + 1 queries: original + k reformulations.
"""

import json


async def get_reformulated_queries(query: str, llm_client, k: int = 3) -> list[str]:
    """
    Generate k semantic variations of the query.
    Returns k+1 queries: original + k reformulations.
    """
    prompt = f"""Generate {k} alternative phrasings of this search query.
Each should capture the same intent but use different words/terminology.

Original query: "{query}"

Return JSON: {{"reformulations": ["alt1", "alt2", "alt3"]}}

Rules:
- Keep same meaning, change wording
- Use synonyms, related terms, different phrasing
- Each should be a standalone search query
- No explanations, just the queries"""

    response = await llm_client.generate(
        prompt=prompt,
        response_format={"type": "json_object"},
        label=f"reformulating query",
    )
    
    # Strip markdown if present
    content = response.content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    
    data = json.loads(content)
    reformulations = data.get("reformulations", [])
    
    # Return original + reformulations
    return [query] + reformulations[:k]
