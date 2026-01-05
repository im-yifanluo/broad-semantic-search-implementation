# Broad Semantic Search Implementation

This repository contains an implementation of the broad semantic search in Asta Paper Finder. More information about Asta Paper Finder can be found at: https://allenai.org/blog/paper-finder

The reason that I implemented the broad semantic search function is because it is technically interesting and challenging. It is the heart of Asta Paper Finder, and I was excited to learn more about it.

## How It Works

1. Query Analysis: LLM extracts the intent, filters, and relevance judgement criteria for the original query. 
2. Query Rephrase: LLM generates 3 rephrasings of the same query
3. Paper Retrieval: A semantic + keyword hybrid approach is used to retrieve papers from Semantic Scholar
4. Aggregation and Deduplication: Snippets are merged and papers are deduplicated.
5. Relavence Judgement: Each paper is scored against the judgement criteria generated in 1.
6. Ranking and Result: A combination of semantic score, citations, and recency is used to rank the results

## Setup
**Install Dependencies**

```bash
pip install -e .
```

**API Key Configuration**
Copy env file and add your Anthropic API key from https://console.anthropic.com/
```bash
cp .env.example .env
```

## Usage 

**Basic Search**
```bash
python3 -m broad_semantic_search.main --query "Your Query"
```

**Limit Results**
```bash
python3 -m broad_semantic_search.main --query "Your Query" --max-results 5
```

**Save to File**
```bash
python3 -m broad_semantic_search.main --query "Your Query" --output results.json
```

## File Structure

```
src/broad_semantic_search/
├── main.py          # CLI entrypoint
├── agent.py         # Pipeline orchestration
├── models.py        # Data classes (Paper, Snippet, JudgedPaper)
├── reformulate.py   # LLM query rephrasing
├── retrieve.py      # Hybrid information retrieval
├── aggregate.py     # Dedupe, snippet merging
├── judge.py         # LLM relevance scoring
├── backends/
│   ├── llm.py       # Anthropic Claude client
│   └── s2.py        # Semantic Scholar API client
└── third_party/
    └── analyze.py   # Query analyzer
```

## API Keys

The `ANTHROPIC_API_KEY` is required, see https://console.anthropic.com/ to generate your key.

`S2_API_KEY` is not required but recommended, see https://www.semanticscholar.org/product/api to generate your key.