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

ROLE_WEIGHTS = {"user": 0.3, "assistant": 0.2, "system": 1.0}
ERROR_KEYWORDS = ["error", "traceback", "failed", "exception", "panic"]


try:
    import tiktoken
    _ENC_CACHE = {}
except ImportError:
    tiktoken = None

def get_tokenizer(model_name="claude-3-5-sonnet-20241022"):
    """Get or cache a tiktoken encoder for a model."""
    if tiktoken is None:
        return None
    if model_name not in _ENC_CACHE:
        try:
            # Fallback for unknown models to cl100k_base (standard for gpt-4)
            _ENC_CACHE[model_name] = tiktoken.encoding_for_model(model_name)
        except (KeyError, ValueError):
            _ENC_CACHE[model_name] = tiktoken.get_encoding("cl100k_base")
    return _ENC_CACHE[model_name]


def estimate_tokens(messages, model="claude-sonnet-4-5-20250929"):
    """Estimate token count for a list of messages.
    
    Uses tiktoken for precise counting if available, otherwise falls
    back to character-based heuristic (~4 chars per token).
    """
    enc = get_tokenizer(model)
    total = 0
    
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        
        # Format content for counting
        text = ""
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        else:
            text = str(content)
            
        if enc:
            # Tiktoken is precise
            # Approximate overhead for message formatting (role + delimiters)
            total += 4 + len(enc.encode(text))
        else:
            # Heuristic fallback
            total += len(text) // 4
            
    # Final conversation overhead (end of sequence)
    if enc:
        total += 3
        
    return max(total, 1)


def _truncate_content(content, max_tokens):
    """Truncate content to approximately max_tokens.

    Handles both plain strings and Anthropic-style content block lists
    (e.g. tool_result with large file contents).
    """
    max_chars = max_tokens * 4
    if isinstance(content, list):
        return _truncate_blocks(content, max_chars)
    content = str(content)
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[...truncated...]\n\n" + content[-half:]


def _truncate_blocks(blocks, max_chars):
    """Truncate Anthropic-style content block lists."""
    total = sum(len(b.get("text", "")) for b in blocks if isinstance(b, dict))
    if total <= max_chars:
        return blocks
    # Budget per block, proportional to total allowance
    result = []
    remaining = max_chars
    for block in blocks:
        if not isinstance(block, dict):
            result.append(block)
            continue
        block = dict(block)
        text = block.get("text", "")
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

    def compress(self, messages, model=None):
        """Compress a list of messages according to the active profile.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Optional model name for context window lookup.

        Returns:
            (compressed_messages, stats_dict)
        """
        if not messages:
            return messages, {"compressed": False, "tokens_before": 0, "tokens_after": 0}

        model = model or "claude-sonnet-4-5-20250929"

        # Truncate individual messages first
        messages = self._truncate_messages(messages)

        tokens_before = estimate_tokens(messages, model=model)

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
            system = self._compress_system(system, max_sys)

        if not old:
            # Nothing to compress in conversation, but system may have been compressed
            result = system + recent
            tokens_after = estimate_tokens(result, model=model)
            if tokens_after < tokens_before:
                stats = {
                    "compressed": True,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "tokens_saved": tokens_before - tokens_after,
                    "ratio": tokens_after / tokens_before if tokens_before > 0 else 1.0,
                    "messages_before": len(messages),
                    "messages_after": len(result),
                }
                if self.stats is not None:
                    self.stats.record(tokens_before, tokens_after)
                return result, stats
            return messages, {
                "compressed": False,
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
            }

        # Score old messages
        scores = []
        for i, m in enumerate(old):
            score = self._score(m)
            # Semantic redundancy check: if too similar to previous, penalize
            if i > 0:
                similarity = _jaccard_similarity(m.get("content", ""), old[i-1].get("content", ""))
                if similarity > 0.8:
                    score *= 0.5  # Heavy penalty for high repetition
            scores.append(score)

        # Apply compression strategy
        if self.profile["method"] == "merge":
            compressed_old = self._merge(old, scores)
        else:
            compressed_old = self._evict(old, scores)

        # Reconstruct
        result = system + compressed_old + recent
        tokens_after = estimate_tokens(result, model=model)

        stats = {
            "compressed": True,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_before - tokens_after,
            "ratio": tokens_after / tokens_before if tokens_before > 0 else 1.0,
            "messages_before": len(messages),
            "messages_after": len(result),
        }

        if self.stats is not None:
            self.stats.record(tokens_before, tokens_after)

        return result, stats

    def _truncate_messages(self, messages):
        """Truncate individual non-system messages that exceed max_msg_tokens.

        System messages are handled separately by _compress_system.
        """
        max_tokens = self.profile["max_msg_tokens"]
        result = []
        for m in messages:
            if m.get("role") == "system":
                result.append(m)
                continue
            content = m.get("content", "")
            if isinstance(content, list):
                tokens = sum(
                    len(b.get("text", "")) for b in content if isinstance(b, dict)
                ) // 4
            else:
                tokens = len(str(content)) // 4
            if tokens > max_tokens:
                m = dict(m)
                m["content"] = _truncate_content(content, max_tokens)
            result.append(m)
        return result

    def _split(self, messages):
        """Split messages into (system, old, recent).

        System messages are always preserved.
        Recent window messages are always preserved.
        Everything else is 'old' and eligible for compression.
        """
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

    def _compress_system(self, system_msgs, max_tokens):
        """Truncate system messages that exceed max_tokens."""
        result = []
        for m in system_msgs:
            content = m.get("content", "")
            tokens = len(str(content)) // 4 if not isinstance(content, list) else (
                sum(len(b.get("text", "")) for b in content if isinstance(b, dict)) // 4
            )
            if tokens > max_tokens:
                m = dict(m)
                m["content"] = _truncate_content(content, max_tokens)
            result.append(m)
        return result

    def _score(self, message):
        """Score a message's importance from 0.0 to 1.0."""
        # Tool-calling safety: always preserve tool messages and calls
        if message.get("role") == "tool" or message.get("tool_calls") or message.get("tool_call_id"):
            return 1.0

        score = 0.0
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        content = str(content)

        # Role weight
        score += ROLE_WEIGHTS.get(message.get("role", ""), 0.1)

        # Code blocks increase importance
        if "```" in content:
            score += 0.3

        # Length: longer messages contribute more context
        tokens = len(content) / 4
        score += min(tokens / 500, 0.2)

        # Error keywords make messages critical
        lower = content.lower()
        if any(kw in lower for kw in ERROR_KEYWORDS):
            score += 0.2

        return min(score, 1.0)

    def _evict(self, old, scores):
        """Remove lowest-scoring messages until keep_ratio is met."""
        keep_count = max(1, int(len(old) * self.profile["keep_ratio"]))
        if keep_count >= len(old):
            return list(old)

        # Pair messages with scores and original index
        indexed = list(enumerate(zip(old, scores)))
        # Sort by score ascending
        indexed.sort(key=lambda x: x[1][1])

        # Mark the lowest-scoring ones for removal
        to_remove = len(old) - keep_count
        remove_indices = {indexed[i][0] for i in range(to_remove)}

        # Build result preserving order, adding summary for removed
        result = []
        removed_count = 0
        for i, m in enumerate(old):
            if i in remove_indices:
                removed_count += 1
            else:
                if removed_count > 0:
                    result.append({
                        "role": "assistant",
                        "content": f"[{removed_count} previous message(s) omitted for brevity]",
                    })
                    removed_count = 0
                result.append(m)

        # Trailing removed messages
        if removed_count > 0:
            result.append({
                "role": "assistant",
                "content": f"[{removed_count} previous message(s) omitted for brevity]",
            })

        return result

    def _merge(self, old, scores):
        """Group old messages into chunks, keep the highest-scoring per chunk."""
        chunk_size = max(2, len(old) // 3)
        chunks = []
        chunk_scores = []
        for i in range(0, len(old), chunk_size):
            chunks.append(old[i : i + chunk_size])
            chunk_scores.append(scores[i : i + chunk_size])

        result = []
        for chunk, c_scores in zip(chunks, chunk_scores):
            # If any message in the chunk is a tool/call, keep the whole chunk verbatim
            has_tool = any(
                m.get("role") == "tool" or m.get("tool_calls") or m.get("tool_call_id")
                for m in chunk
            )
            if len(chunk) == 1 or has_tool:
                result.extend(chunk)
                continue

            # Keep the highest-scoring message from the chunk verbatim
            best_idx = max(range(len(c_scores)), key=lambda j: c_scores[j])
            best_msg = chunk[best_idx]

            # Summarize the rest
            rest_count = len(chunk) - 1
            snippets = []
            for j, m in enumerate(chunk):
                if j == best_idx:
                    continue
                text = str(m.get("content", ""))
                if isinstance(m.get("content"), list):
                    text = " ".join(
                        b.get("text", "")[:80] for b in m["content"] if isinstance(b, dict)
                    )
                snippets.append(f"  - [{m.get('role', '?')}] {text[:120]}")

            summary_content = (
                f"[Merged {rest_count} other message(s) from this section]\n"
                + "\n".join(snippets[:5])
            )
            if len(snippets) > 5:
                summary_content += f"\n  ... and {len(snippets) - 5} more"

            result.append(best_msg)
            result.append({"role": "assistant", "content": summary_content})

        return result
