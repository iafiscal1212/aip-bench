"""OpenAI API provider adapter (/v1/chat/completions)."""

from .base import Provider


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
