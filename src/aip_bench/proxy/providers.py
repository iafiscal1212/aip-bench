"""
Provider detection and request/response adapters.

Auto-detects OpenAI, Anthropic, or generic providers based on request
path and headers, then normalizes message extraction and forwarding.
"""


class Provider:
    """Base provider adapter."""

    name = "generic"

    def extract_messages(self, body):
        """Extract messages list from request body."""
        return body.get("messages", [])

    def replace_messages(self, body, messages):
        """Return body with messages replaced."""
        body = dict(body)
        body["messages"] = messages
        return body

    def forward_headers(self, headers):
        """Build headers to forward to upstream API."""
        forwarded = {}
        for key, value in headers.items():
            lower = key.lower()
            # Skip hop-by-hop headers
            if lower in (
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
            ):
                continue
            forwarded[key] = value
        return forwarded

    def build_url(self, target, path):
        """Build the full upstream URL."""
        if target:
            return target.rstrip("/") + path
        return None


class AnthropicProvider(Provider):
    """Adapter for Anthropic API (/v1/messages)."""

    name = "anthropic"

    def extract_messages(self, body):
        return body.get("messages", [])

    def replace_messages(self, body, messages):
        body = dict(body)
        body["messages"] = messages
        return body

    def forward_headers(self, headers):
        forwarded = {}
        for key, value in headers.items():
            lower = key.lower()
            if lower in ("host", "content-length", "transfer-encoding", "connection"):
                continue
            forwarded[key] = value
        # Ensure anthropic-version is forwarded
        return forwarded

    def build_url(self, target, path):
        base = target or "https://api.anthropic.com"
        return base.rstrip("/") + path


class OpenAIProvider(Provider):
    """Adapter for OpenAI API (/v1/chat/completions)."""

    name = "openai"

    def extract_messages(self, body):
        return body.get("messages", [])

    def replace_messages(self, body, messages):
        body = dict(body)
        body["messages"] = messages
        return body

    def build_url(self, target, path):
        base = target or "https://api.openai.com"
        return base.rstrip("/") + path


class GenericProvider(Provider):
    """Fallback provider — passthrough with optional message compression."""

    name = "generic"

    def build_url(self, target, path):
        if target:
            return target.rstrip("/") + path
        # Generic requires explicit target
        return None


def detect_provider(path, headers=None):
    """Detect provider from request path and headers.

    Args:
        path: Request URL path (e.g., '/v1/messages').
        headers: Optional dict of request headers.

    Returns:
        Provider instance (AnthropicProvider, OpenAIProvider, or GenericProvider).
    """
    if "/messages" in path and "/chat/completions" not in path:
        return AnthropicProvider()
    elif "/chat/completions" in path:
        return OpenAIProvider()
    return GenericProvider()
