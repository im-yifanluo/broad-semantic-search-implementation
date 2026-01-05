""" 
CLI Entrypoint for Broad Semantic Search

Runs the full pipeline on a user query and outputs results as JSON.
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

# Load .env file before importing modules that need env vars
load_dotenv()

from broad_semantic_search.agent import BroadSearchAgent
from broad_semantic_search.backends.llm import LLMClient
from broad_semantic_search.backends.s2 import S2Client

# Parse command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Broad Semantic Search - Find relevant papers using LLM-powered search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The search query describing papers you're looking for",
    )
    
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=20,
        help="Maximum number of results to return (default: 20)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    
    return parser.parse_args()

# Main entry point of the entire program
async def main() -> int:
    args = parse_args()
    
    # Initialize LLM client (reads ANTHROPIC_API_KEY from environment)
    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm_client = LLMClient()
        print("LLM client initialized", file=sys.stderr)
    else:
        print("Warning: ANTHROPIC_API_KEY not set, running without query analysis", file=sys.stderr)
    
    # Initialize S2 client (reads S2_API_KEY from environment, optional)
    s2_client = S2Client()  # Works without API key but with rate limits
    if os.environ.get("S2_API_KEY"):
        print("S2 client initialized with API key", file=sys.stderr)
    else:
        print("S2 client initialized (no API key - rate limited)", file=sys.stderr)
    
    agent = BroadSearchAgent(
        llm_client=llm_client,
        s2_client=s2_client,
        max_results=args.max_results,
    )
    
    # Run search
    result = await agent.run(query=args.query)
    
    # Format output as JSON
    output = json.dumps(result, indent=2, ensure_ascii=False)
    
    # Write output
    if args.output: # Write to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else: # Print to stdout
        print(output)
    
    return 0

# Just some standard boilerplate
if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
    