"""
aip_bench.proxy — Accordion-powered LLM proxy.

Compresses chat messages in real-time between any LLM client and API provider.

Quick start:
    # CLI
    aip-proxy --port 8080 --profile balanced

    # Python
    from aip_bench.proxy import MessageAccordion

    accordion = MessageAccordion(profile="balanced")
    compressed, stats = accordion.compress(messages, model="gpt-4o")
"""

from .accordion import (
    MessageAccordion,
    estimate_tokens,
    PROFILES,
)
from .providers import (
    detect_provider,
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    GenericProvider,
)
from .stats import CompressionStats
from .cli import main

__all__ = [
    "MessageAccordion",
    "estimate_tokens",
    "PROFILES",
    "detect_provider",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GenericProvider",
    "CompressionStats",
    "main",
]
