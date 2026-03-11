"""Base provider adapter."""


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

    def count_tokens(self, messages, model=None):
        """Count tokens in a list of messages.
        
        Default implementation uses a simple character-based heuristic.
        Subclasses should override for model-specific accuracy.
        """
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                )
            else:
                text = str(content)
            total += len(text) // 4
        return max(total, 1)

    def build_url(self, target, path):
        """Build the full upstream URL."""
        if target:
            return target.rstrip("/") + path
        return None
