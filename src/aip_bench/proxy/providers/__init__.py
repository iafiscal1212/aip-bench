"""
Provider detection and request/response adapters.

Auto-detects OpenAI, Anthropic, or generic providers based on request
path and headers, then normalizes message extraction and forwarding.
Each provider lives in its own module and is lazily imported on first use.
"""


def detect_provider(path, headers=None):
    """Detect provider from request path and headers.

    Args:
        path: Request URL path (e.g., '/v1/messages').
        headers: Optional dict of request headers.

    Returns:
        Provider instance (AnthropicProvider, OpenAIProvider, or GenericProvider).
    """
    if "/messages" in path and "/chat/completions" not in path:
        from .anthropic import AnthropicProvider
        return AnthropicProvider()
    elif "/chat/completions" in path:
        from .openai import OpenAIProvider
        return OpenAIProvider()
    from .generic import GenericProvider
    return GenericProvider()


# Re-exports for backward compatibility
from .base import Provider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .generic import GenericProvider

__all__ = [
    "detect_provider",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GenericProvider",
]
