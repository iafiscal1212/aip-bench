"""Anthropic API provider adapter (/v1/messages)."""

from .base import Provider


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

    def _filter_empty_user_messages(self, messages):
        """Removes user messages with empty, absent, or empty list content."""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if content is None or (isinstance(content, list) and not content) or \
                   (isinstance(content, str) and not content.strip()):
                    continue
            filtered_messages.append(msg)
        return filtered_messages

    def _validate_tool_use(self, messages):
        """Filter tool_result blocks to ensure they have a corresponding tool_use ID
        from the immediately preceding assistant message.
        """
        if not messages:
            return []

        validated_messages = [messages[0]] if messages else []

        for i in range(1, len(messages)):
            curr_msg = messages[i]
            prev_msg = messages[i - 1]

            processed_curr_msg = curr_msg

            if (curr_msg.get("role") == "user" and isinstance(curr_msg.get("content"), list) and
                    prev_msg.get("role") == "assistant"):

                valid_ids = set()
                if isinstance(prev_msg.get("content"), list):
                    valid_ids = self._get_valid_tool_ids(prev_msg["content"])

                filtered_content = self._filter_tool_results(curr_msg["content"], valid_ids)

                if len(filtered_content) < len(curr_msg["content"]):
                    processed_curr_msg = dict(curr_msg)
                    processed_curr_msg["content"] = filtered_content

            validated_messages.append(processed_curr_msg)

        return validated_messages

    def extract_messages(self, body):
        return body.get("messages", [])

    def replace_messages(self, body, messages):
        body = dict(body)
        messages = self._filter_empty_user_messages(messages)
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
        return forwarded

    def build_url(self, target, path):
        base = target or "https://api.anthropic.com"
        return base.rstrip("/") + path
