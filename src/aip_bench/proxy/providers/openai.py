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

    def count_tokens(self, messages, model=None):
        """Count tokens precisely using tiktoken."""
        try:
            import tiktoken
        except ImportError:
            return super().count_tokens(messages, model)
            
        model = model or "gpt-4o"
        try:
            enc = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            enc = tiktoken.get_encoding("cl100k_base")
            
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                )
            else:
                text = str(content)
            # 4 tokens overhead per message (role + content wrapper)
            total += 4 + len(enc.encode(text))
        # 3 tokens overhead for the conversation end
        return total + 3

    def build_url(self, target, path):
        base = target or "https://api.openai.com"
        return base.rstrip("/") + path
