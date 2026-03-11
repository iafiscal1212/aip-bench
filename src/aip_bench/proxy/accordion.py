"""
MessageAccordion: message compression engine for LLM proxy.

Implements the Accordion pattern (batch -> flush -> compress) for chat messages.
Scores messages by importance, then evicts or merges low-value ones to reduce
input token count while preserving conversation quality.
"""

# Context window sizes for known models (tokens)
MODEL_CONTEXTS = {
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "llama3": 8_192,
    "mistral": 32_000,
}

PROFILES = {
    "conservative": {
        "min_tokens": 8_000,    # Compress when >8K tokens (absolute)
        "keep_ratio": 0.85,
        "recent_window": 20,
        "method": "evict",
        "max_msg_tokens": 8192,
        "max_system_tokens": 0, # 0 = no system compression
    },
    "balanced": {
        "min_tokens": 5_000,    # Compress when >5K tokens
        "keep_ratio": 0.60,
        "recent_window": 12,
        "method": "evict",
        "max_msg_tokens": 4096,
        "max_system_tokens": 4096,
    },
    "aggressive": {
        "min_tokens": 2_000,    # Compress when >2K tokens
        "keep_ratio": 0.35,
        "recent_window": 6,
        "method": "merge",
        "max_msg_tokens": 2048,
        "max_system_tokens": 2048,
    },
}

ROLE_WEIGHTS = {"user": 0.5, "assistant": 0.2, "system": 1.0}
ERROR_KEYWORDS = ["error", "traceback", "failed", "exception", "panic"]


def _jaccard_similarity(text1, text2):
    """Compute token-based Jaccard similarity between two strings."""
    s1 = set(str(text1).lower().split())
    s2 = set(str(text2).lower().split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


class MessageAccordion:
    """Compress chat messages using the Accordion pattern.

    Accumulates messages, scores them by importance, and compresses
    by evicting or merging low-value messages while preserving recent
    context and system prompts.
    """

    def __init__(self, profile="balanced"):
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: {list(PROFILES.keys())}"
            )
        self.profile = PROFILES[profile]
        self.profile_name = profile
        self.stats = None  # Set externally or via CompressionStats

    def compress(self, messages, provider, model=None):
        """Compress a list of messages according to the active profile.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            provider: Provider instance for token counting (Point 1).
            model: Optional model name for context window lookup.

        Returns:
            (compressed_messages, stats_dict)
        """
        if not messages:
            return messages, {"compressed": False, "tokens_before": 0, "tokens_after": 0}

        model = model or "claude-3-5-sonnet-20241022"

        # 1. Truncate individual messages using unified provider counting (Point 2)
        messages = self._truncate_messages(messages, provider, model)

        tokens_before = provider.count_tokens(messages, model)

        # Check absolute token threshold
        if tokens_before < self.profile["min_tokens"]:
            return messages, {
                "compressed": False,
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
            }

        # Split into system, old, recent
        system, old, recent = self._split(messages)

        # Compress system prompt if configured and large
        max_sys = self.profile.get("max_system_tokens", 0)
        if max_sys > 0:
            system = self._compress_system(system, provider, model, max_sys)

        if not old:
            # Nothing to compress in conversation, but system may have been compressed
            result = system + recent
            tokens_after = provider.count_tokens(result, model)
            if tokens_after < tokens_before:
                stats = self._build_stats(tokens_before, tokens_after, len(messages), len(result))
                return result, stats
            return messages, {
                "compressed": False,
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
            }

        # 2. Score old messages with role-specific redundancy check (Point 3)
        scores = []
        last_seen = {}  # Store last message content per role
        
        for m in old:
            role = m.get("role")
            score = self._score(m)
            
            # Semantic redundancy check (Point 3: detect ping-pong patterns)
            current_content = self._get_text_content(m)
            if role in last_seen:
                similarity = _jaccard_similarity(current_content, last_seen[role])
                if similarity > 0.8:
                    score *= 0.5  # Heavy penalty for repetition within role
            
            last_seen[role] = current_content
            scores.append(score)

        # Apply compression strategy
        if self.profile["method"] == "merge":
            compressed_old = self._merge(old, scores)
        else:
            compressed_old = self._evict(old, scores)

        # Reconstruct
        result = system + compressed_old + recent
        tokens_after = provider.count_tokens(result, model)

        stats = self._build_stats(tokens_before, tokens_after, len(messages), len(result))
        return result, stats

    def _truncate_messages(self, messages, provider, model):
        """Truncate individual non-system messages using unified counting (Point 2)."""
        max_tokens = self.profile["max_msg_tokens"]
        result = []
        for m in messages:
            if m.get("role") == "system":
                result.append(m)
                continue
            
            # Use provider's exact count for this message alone
            tokens = provider.count_tokens([m], model)
            if tokens > max_tokens:
                m = dict(m)
                m["content"] = self._do_truncate(m.get("content", ""), max_tokens)
            result.append(m)
        return result

    def _do_truncate(self, content, max_tokens):
        """Truncate content to approximately max_tokens (approx 4 chars/token)."""
        max_chars = max_tokens * 4
        if isinstance(content, list):
            return self._truncate_blocks(content, max_chars)
        content = str(content)
        if len(content) <= max_chars:
            return content
        half = max_chars // 2
        return content[:half] + "\n\n[...truncated...]\n\n" + content[-half:]

    def _split(self, messages):
        system = []
        rest = []
        for m in messages:
            if m.get("role") == "system":
                system.append(m)
            else:
                rest.append(m)

        window = self.profile["recent_window"]
        if len(rest) <= window:
            return system, [], rest

        old = rest[:-window]
        recent = rest[-window:]
        return system, old, recent

    def _compress_system(self, system_msgs, provider, model, max_tokens):
        """Truncate system messages using unified counting (Point 2)."""
        result = []
        for m in system_msgs:
            if provider.count_tokens([m], model) > max_tokens:
                m = dict(m)
                m["content"] = self._do_truncate(m.get("content", ""), max_tokens)
            result.append(m)
        return result

    def _score(self, message):
        """Score message importance (Point 5: higher user importance)."""
        if message.get("role") == "tool" or message.get("tool_calls") or message.get("tool_call_id"):
            return 1.0

        score = 0.0
        content = self._get_text_content(message)

        # Role weight (Point 5: User gets a floor of 0.5)
        score += ROLE_WEIGHTS.get(message.get("role", ""), 0.1)

        # Code blocks increase importance
        if "```" in content:
            score += 0.3

        # Length bonus
        score += min(len(content) / 2000, 0.2)

        # Error keywords
        if any(kw in content.lower() for kw in ERROR_KEYWORDS):
            score += 0.2

        return min(score, 1.0)

    def _get_text_content(self, message):
        """Flatten message content to string."""
        content = message.get("content", "")
        if isinstance(content, list):
            return " ".join(
                str(b.get("text", "")) for b in content if isinstance(b, dict)
            )
        return str(content)

    def _evict(self, old, scores):
        """Remove lowest-scoring messages preserving order (Point 4: API-safe role)."""
        keep_count = max(1, int(len(old) * self.profile["keep_ratio"]))
        if keep_count >= len(old):
            return list(old)

        indexed = list(enumerate(zip(old, scores)))
        indexed.sort(key=lambda x: x[1][1])

        to_remove = len(old) - keep_count
        remove_indices = {indexed[i][0] for i in range(to_remove)}

        result = []
        removed_count = 0
        for i, m in enumerate(old):
            if i in remove_indices:
                removed_count += 1
            else:
                if removed_count > 0:
                    # Using 'user' role with bracketed prefix for maximum API compatibility
                    result.append({
                        "role": "user",
                        "content": f"[CONTEXT: {removed_count} message(s) omitted for brevity]",
                    })
                    removed_count = 0
                result.append(m)

        if removed_count > 0:
            result.append({
                "role": "user", 
                "content": f"[CONTEXT: {removed_count} message(s) omitted for brevity]",
            })
        return result

    def _merge(self, old, scores):
        """Merge old messages, preserving high-scoring ones (Point 4: API-safe role)."""
        chunk_size = max(2, len(old) // 3)
        chunks = []
        chunk_scores = []
        for i in range(0, len(old), chunk_size):
            chunks.append(old[i : i + chunk_size])
            chunk_scores.append(scores[i : i + chunk_size])

        result = []
        for chunk, c_scores in zip(chunks, chunk_scores):
            # If any message in the chunk is a tool/call, keep verbatim (atomic)
            has_tool = any(
                m.get("role") == "tool" or m.get("tool_calls") or m.get("tool_call_id")
                for m in chunk
            )
            if len(chunk) == 1 or has_tool:
                result.extend(chunk)
                continue

            best_idx = max(range(len(c_scores)), key=lambda j: c_scores[j])
            best_msg = chunk[best_idx]

            # Using 'user' role for summary to avoid mid-conversation system role errors
            rest_count = len(chunk) - 1
            summary_content = f"[CONTEXT: {rest_count} previous message(s) merged for brevity here]"
            
            result.append(best_msg)
            result.append({"role": "user", "content": summary_content})

        return result

    def _build_stats(self, before, after, n_before, n_after):
        """Build stats dict, only marking compressed=True if tokens were actually saved."""
        compressed = after < before
        stats = {
            "compressed": compressed,
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": max(0, before - after),
            "ratio": after / before if before > 0 else 1.0,
            "messages_before": n_before,
            "messages_after": n_after,
        }
        if self.stats and compressed:
            self.stats.record(before, after)
        return stats

    def _truncate_blocks(self, blocks, max_chars):
        """Truncate Anthropic-style content block lists."""
        total = sum(len(str(b.get("text", ""))) for b in blocks if isinstance(b, dict))
        if total <= max_chars:
            return blocks
        result = []
        remaining = max_chars
        for block in blocks:
            if not isinstance(block, dict):
                result.append(block)
                continue
            block = dict(block)
            text = str(block.get("text", ""))
            if not text:
                result.append(block)
                continue
            budget = max(200, remaining * len(text) // max(total, 1))
            if len(text) > budget:
                half = budget // 2
                block["text"] = text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
            remaining -= len(block["text"])
            result.append(block)
        return result
