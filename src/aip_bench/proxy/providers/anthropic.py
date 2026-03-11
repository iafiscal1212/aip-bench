"""Anthropic API provider adapter (/v1/messages)."""

import logging

from .base import Provider

logger = logging.getLogger("aip-proxy.anthropic")


class AnthropicProvider(Provider):
    """Adapter for Anthropic API (/v1/messages)."""

    name = "anthropic"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_valid_tool_ids(self, content):
        """Extract valid tool_use IDs from an assistant message's content."""
        return {
            block.get("id") for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    def _filter_tool_results(self, content, valid_ids):
        """Keep non-tool_result blocks, and tool_result blocks with valid IDs only.
        If valid_ids is empty, all tool_result blocks are removed.
        """
        return [
            block for block in content
            if not (isinstance(block, dict) and block.get("type") == "tool_result")
            or block.get("tool_use_id") in valid_ids
        ]

    # ------------------------------------------------------------------
    # Normalization steps
    # ------------------------------------------------------------------

    def _normalize_system_prompt(self, messages, body):
        """Extract system-role messages from messages array into top-level body['system'].

        Anthropic API does not accept role='system' inside the messages array.
        Clients using an OpenAI-style format may send it there; this fixes that.

        Returns (filtered_messages, updated_body).
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if not system_msgs:
            return messages, body

        # Build extracted system text
        parts = []
        for m in system_msgs:
            content = m.get("content", "")
            if isinstance(content, list):
                text = "\n\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)
            if text:
                parts.append(text)

        extracted = "\n\n".join(parts)
        logger.debug(
            "Extracted %d system-role message(s) from messages array → top-level system field",
            len(system_msgs),
        )

        body = dict(body)
        existing = body.get("system")
        if existing:
            # Prepend extracted system content to existing system field
            if isinstance(existing, str):
                body["system"] = extracted + "\n\n" + existing if extracted else existing
            elif isinstance(existing, list):
                # Keep as list, prepend as text block
                body["system"] = (
                    [{"type": "text", "text": extracted}] + existing
                    if extracted else existing
                )
        else:
            body["system"] = extracted

        filtered = [m for m in messages if m.get("role") != "system"]
        return filtered, body

    def _validate_tool_use(self, messages):
        """Filter tool_result blocks to ensure they reference a tool_use ID from
        the immediately preceding assistant message. Removes orphaned results.
        """
        if not messages:
            return []

        validated = [messages[0]]

        for i in range(1, len(messages)):
            curr = messages[i]
            prev = messages[i - 1]
            processed = curr

            if (
                curr.get("role") == "user"
                and isinstance(curr.get("content"), list)
                and prev.get("role") == "assistant"
            ):
                valid_ids = set()
                if isinstance(prev.get("content"), list):
                    valid_ids = self._get_valid_tool_ids(prev["content"])

                filtered_content = self._filter_tool_results(curr["content"], valid_ids)

                if len(filtered_content) != len(curr["content"]):
                    removed = len(curr["content"]) - len(filtered_content)
                    logger.debug(
                        "Removed %d orphaned tool_result block(s) from user message at index %d "
                        "(valid tool_use IDs: %s)",
                        removed, i, valid_ids or "none",
                    )
                    processed = dict(curr)
                    processed["content"] = filtered_content

            validated.append(processed)

        return validated

    def _filter_empty_user_messages(self, messages):
        """Remove user messages with no content (None, empty string, or empty list)."""
        result = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if (
                    content is None
                    or (isinstance(content, list) and not content)
                    or (isinstance(content, str) and not content.strip())
                ):
                    logger.debug("Removed empty user message")
                    continue
            result.append(msg)
        return result

    def _merge_consecutive_roles(self, messages):
        """Merge consecutive messages that share the same role.

        Anthropic accepts them (it merges them server-side), but doing it
        explicitly avoids surprises with tool_use/tool_result ordering.
        Logs a WARNING because this indicates the accordion produced an
        invalid sequence (e.g. two assistant messages after eviction).
        """
        if not messages:
            return messages

        result = [dict(messages[0])]
        for msg in messages[1:]:
            prev = result[-1]
            if msg.get("role") == prev.get("role"):
                logger.warning(
                    "Merging consecutive %r messages — accordion may have created invalid sequence",
                    msg.get("role"),
                )
                prev["content"] = self._merge_content(prev.get("content"), msg.get("content"))
            else:
                result.append(dict(msg))
        return result

    @staticmethod
    def _merge_content(a, b):
        """Merge two message content values into one."""
        def to_blocks(c):
            if isinstance(c, list):
                return c
            return [{"type": "text", "text": str(c) if c is not None else ""}]

        if isinstance(a, str) and isinstance(b, str):
            return a + "\n\n" + b

        # At least one is a list (or None)
        blocks_a = to_blocks(a) if a is not None else []
        blocks_b = to_blocks(b) if b is not None else []
        return blocks_a + blocks_b

    def _ensure_starts_with_user(self, messages):
        """Ensure the first message is role='user', as required by Anthropic.

        Instead of dropping leading assistant messages (losing context), injects
        a minimal placeholder user turn so the assistant content is preserved.
        This is the standard approach used by production proxies.
        """
        if not messages or messages[0].get("role") == "user":
            return messages

        logger.warning(
            "Conversation starts with role=%r; injecting placeholder user turn "
            "to satisfy Anthropic's user-first requirement",
            messages[0].get("role"),
        )
        placeholder = {"role": "user", "content": "[Context omitted]"}
        return [placeholder] + list(messages)

    def count_tokens(self, messages, model=None):
        """Estimate Anthropic tokens using tiktoken + correction factor.
        
        Anthropic's tokenizer is similar to OpenAI's cl100k_base but 
        typically counts ~10% more tokens for the same text.
        """
        try:
            import tiktoken
        except ImportError:
            return super().count_tokens(messages, model)
            
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
            # Claude overhead is slightly different but ~4 tokens per msg is a good proxy
            total += 4 + int(len(enc.encode(text)) * 1.1)
        return total + 3

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def extract_messages(self, body):
        return body.get("messages", [])

    def replace_messages(self, body, messages):
        body = dict(body)
        n0 = len(messages)
        logger.debug("replace_messages: start with %d messages", n0)

        # 1. Move system-role messages out of the messages array (Bug 3)
        messages, body = self._normalize_system_prompt(messages, body)
        logger.debug("replace_messages: after system normalization → %d messages", len(messages))

        # 2. Merge consecutive same-role messages first, so that validate_tool_use
        #    sees the correct predecessor when the accordion split an assistant turn
        #    (e.g. tool_use block + summary placeholder → one merged message).
        messages = self._merge_consecutive_roles(messages)
        logger.debug("replace_messages: after role merge → %d messages", len(messages))

        # 3. Remove orphaned tool_results (must run after merge so consecutive
        #    assistant turns are already unified before checking tool_use IDs).
        messages = self._validate_tool_use(messages)
        logger.debug("replace_messages: after tool_use validation → %d messages", len(messages))

        # 4. Remove user messages emptied by step 3.
        messages = self._filter_empty_user_messages(messages)
        logger.debug("replace_messages: after empty-user filter → %d messages", len(messages))

        # 5. Ensure conversation starts with a user message (Bug 4)
        messages = self._ensure_starts_with_user(messages)
        logger.debug("replace_messages: after ensure-starts-with-user → %d messages", len(messages))

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
