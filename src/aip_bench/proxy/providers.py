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

    def _get_valid_tool_ids(self, content):
            """Extract valid tool_use IDs from an assistant message's content."""
            return {
                block.get("id") for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
            }

    def _filter_tool_results(self, content, valid_ids):
        """Keep blocks that are NOT tool_results, or tool_results with valid IDs.
        If valid_ids is empty, all tool_result blocks are removed.
        """
        return [
            block for block in content
            if not (isinstance(block, dict) and block.get("type") == "tool_result")
            or (isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in valid_ids)
        ]

    def _validate_tool_use(self, messages):
        """Filter tool_result blocks to ensure they have a corresponding tool_use ID
        from the immediately preceding assistant message.
        """
        if not messages:
            return []

        # We need to build the validated list iteratively
        validated_messages = [messages[0]] if messages else []

        for i in range(1, len(messages)):
            curr_msg = messages[i]
            prev_msg = messages[i-1]

            processed_curr_msg = curr_msg # Default to keeping the original message

            # We only care about user messages that follow assistant messages
            if (curr_msg.get("role") == "user" and isinstance(curr_msg.get("content"), list) and
                prev_msg.get("role") == "assistant"): # Removed isinstance(prev_msg.get("content"), list) here

                valid_ids = set() # Initialize valid_ids to empty set
                if isinstance(prev_msg.get("content"), list): # Only try to get IDs if content is a list
                    valid_ids = self._get_valid_tool_ids(prev_msg["content"])

                filtered_content = self._filter_tool_results(curr_msg["content"], valid_ids)

                # If the content changed, create a new message dict
                if len(filtered_content) < len(curr_msg["content"]):
                    processed_curr_msg = dict(curr_msg)
                    processed_curr_msg["content"] = filtered_content

            validated_messages.append(processed_curr_msg)

        return validated_messages

    def extract_messages(self, body):
        return body.get("messages", [])

    def replace_messages(self, body, messages):
        body = dict(body)
        # Validate tool use before sending to Anthropic
        messages = self._validate_tool_use(messages)
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
