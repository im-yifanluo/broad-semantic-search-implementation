"""
LLM client for Anthropic Claude.

Uses httpx for async HTTP requests.
"""

import logging
import os
import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Minimal Claude API wrapper."""
    
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
    
    async def generate(self, prompt: str, response_format: dict = None, label: str = None) -> "LLMResponse":
        """Call Claude and return response."""
        # Log what we're doing
        if label:
            logger.info(f"LLM: {label}")
        else:
            prompt_preview = prompt[:60].replace('\n', ' ') + "..." if len(prompt) > 60 else prompt
            logger.info(f"LLM call: {prompt_preview}")
        
        system = None
        if response_format and response_format.get("type") == "json_object":
            system = "Respond with valid JSON only."
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            r.raise_for_status()
            data = r.json()
        
        content = data["content"][0]["text"]
        logger.info(f"LLM response: {len(content)} chars")
        return LLMResponse(content=content)


class LLMResponse:
    """Response from LLM."""
    def __init__(self, content: str):
        self.content = content
