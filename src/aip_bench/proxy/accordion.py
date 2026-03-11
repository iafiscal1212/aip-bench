"""
MessageAccordion: message compression engine for LLM proxy.

Implements the Accordion pattern (batch -> flush -> compress) for chat messages.
Groups messages into atomic units (user/assistant pairs or tool-call/tool-result
sequences), scores atoms by importance, then silently evicts or merges low-value
atoms to reduce input token count while preserving conversation quality and
strict role alternation required by Anthropic and OpenAI APIs.
"""

PROFILES = {
    "conservative": {
        "min_tokens": 8_000,
        "keep_ratio": 0.85,
        "recent_window": 20,
        "method": "evict",
        "max_msg_tokens": 8192,
        "max_system_tokens": 0,
    },
    "balanced": {
        "min_tokens": 5_000,
        "keep_ratio": 0.60,
        "recent_window": 12,
        "method": "evict",
        "max_msg_tokens": 4096,
        "max_system_tokens": 4096,
    },
    "aggressive": {
        "min_tokens": 2_000,
        "keep_ratio": 0.35,
        "recent_window": 6,
        "method": "merge",
        "max_msg_tokens": 2048,
        "max_system_tokens": 2048,
    },
}

ROLE_WEIGHTS = {"user": 0.5, "assistant": 0.2, "system": 1.0}
ERROR_KEYWORDS = ["error", "traceback", "failed", "exception", "panic"]


def estimate_tokens(messages):
    """Estimate token count for a list of messages (~4 chars per token)."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
            total += len(text) // 4
        else:
            total += len(str(content)) // 4
    return max(total, 1)


def _jaccard_similarity(text1, text2):
    """Compute token-based Jaccard similarity between two strings."""
    s1 = set(str(text1).lower().split())
    s2 = set(str(text2).lower().split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _is_tool_message(m):
    """Return True if `m` is part of a tool-call/tool-result exchange."""
    if m.get("role") == "tool" or m.get("tool_calls") or m.get("tool_call_id"):
        return True
    content = m.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                return True
    return False


class MessageAccordion:
    """Compress chat messages using the Accordion pattern.

    Groups messages into atomic units, scores them by importance, and
    silently evicts or merges low-value atoms while preserving recent
    context, system prompts, and strict role alternation.
    """

    def __init__(self, profile="balanced"):
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: {list(PROFILES.keys())}"
            )
        self.profile = PROFILES[profile]
        self.profile_name = profile
        self.stats = None  # Set externally or via CompressionStats

    # ------------------------------------------------------------------
    # Token counting helper
    # ------------------------------------------------------------------

    def _count_tokens(self, messages, provider, model):
        """Count tokens using provider if available, else estimate."""
        if provider is not None:
            return provider.count_tokens(messages, model)
        return estimate_tokens(messages)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, messages, provider=None, model=None):
        """Compress a list of messages according to the active profile.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            provider: Optional provider instance for accurate token counting.
            model: Optional model name.

        Returns:
            (compressed_messages, stats_dict)
        """
        if not messages:
            return messages, {"compressed": False, "tokens_before": 0, "tokens_after": 0}

        model = model or "claude-3-5-sonnet-20241022"

        # 1. Truncate individual messages
        messages = self._truncate_messages(messages, provider, model)

        tokens_before = self._count_tokens(messages, provider, model)

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
            result = system + recent
            tokens_after = self._count_tokens(result, provider, model)
            if tokens_after < tokens_before:
                stats = self._build_stats(tokens_before, tokens_after, len(messages), len(result))
                return result, stats
            return messages, {
                "compressed": False,
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
            }

        # 2. Build atoms from old messages
        atoms = self._make_atoms(old)

        # 3. Score atoms with role-specific redundancy check
        atom_scores = []
        last_seen = {}
        for atom in atoms:
            score = self._score_atom(atom)
            # Semantic redundancy: compare representative text per role
            for m in atom:
                role = m.get("role")
                current_content = self._get_text_content(m)
                if role in last_seen:
                    similarity = _jaccard_similarity(current_content, last_seen[role])
                    if similarity > 0.8:
                        score *= 0.5
                        break
                last_seen[role] = current_content
            atom_scores.append(score)

        # 4. Apply compression strategy (operates on atoms)
        if self.profile["method"] == "merge":
            compressed_atoms = self._merge(atoms, atom_scores)
        else:
            compressed_atoms = self._evict(atoms, atom_scores)

        # 5. Flatten atoms back to messages
        compressed_old = [m for atom in compressed_atoms for m in atom]

        # 6. Consolidate consecutive same-role messages
        compressed_old = self._consolidate_roles(compressed_old)

        # Reconstruct
        result = system + compressed_old + recent
        tokens_after = self._count_tokens(result, provider, model)

        n_result = len(result)
        stats = self._build_stats(tokens_before, tokens_after, len(messages), n_result)
        return result, stats

    # ------------------------------------------------------------------
    # Atom construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_atoms(messages):
        """Group a flat list of messages into atomic units.

        * A tool atom: [assistant(tool_calls/tool_use), tool/user(tool_result)]
        * A normal atom: [user, assistant] pair (or single message if unpaired)

        Atoms are indivisible — eviction/merge operates on whole atoms.
        """
        atoms = []
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            # Detect tool-call initiator (assistant with tool_calls or tool_use blocks)
            if msg.get("role") == "assistant" and _is_tool_message(msg):
                atom = [msg]
                i += 1
                # Collect all subsequent tool-response messages
                while i < n and _is_tool_message(messages[i]) and messages[i].get("role") != "assistant":
                    atom.append(messages[i])
                    i += 1
                atoms.append(atom)
                continue

            # Normal pair: try to group [user, assistant]
            if msg.get("role") == "user" and i + 1 < n and messages[i + 1].get("role") == "assistant":
                # Only pair if neither is a tool message
                if not _is_tool_message(msg) and not _is_tool_message(messages[i + 1]):
                    atoms.append([msg, messages[i + 1]])
                    i += 2
                    continue

            # Single message atom (fallback)
            atoms.append([msg])
            i += 1

        return atoms

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_atom(self, atom):
        """Score an atom's importance (max score of its messages)."""
        return max(self._score(m) for m in atom)

    def _score(self, message):
        """Score message importance."""
        if _is_tool_message(message):
            return 1.0

        score = 0.0
        content = self._get_text_content(message)

        score += ROLE_WEIGHTS.get(message.get("role", ""), 0.1)

        if "```" in content:
            score += 0.3

        score += min(len(content) / 2000, 0.2)

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

    # ------------------------------------------------------------------
    # Eviction (silent — no markers injected)
    # ------------------------------------------------------------------

    def _evict(self, atoms, scores):
        """Silently remove lowest-scoring atoms, preserving order."""
        keep_count = max(1, int(len(atoms) * self.profile["keep_ratio"]))
        if keep_count >= len(atoms):
            return list(atoms)

        indexed = sorted(enumerate(scores), key=lambda x: x[1])
        to_remove = len(atoms) - keep_count
        remove_indices = {indexed[i][0] for i in range(to_remove)}

        return [atom for i, atom in enumerate(atoms) if i not in remove_indices]

    # ------------------------------------------------------------------
    # Merge (silent — no markers injected)
    # ------------------------------------------------------------------

    def _merge(self, atoms, scores):
        """Keep only highest-scoring atoms per chunk, silently dropping rest."""
        chunk_size = max(2, len(atoms) // 3)
        result = []
        for i in range(0, len(atoms), chunk_size):
            chunk = atoms[i : i + chunk_size]
            c_scores = scores[i : i + chunk_size]

            if len(chunk) == 1:
                result.extend(chunk)
                continue

            # Keep the best-scoring atom, silently drop the rest
            best_idx = max(range(len(c_scores)), key=lambda j: c_scores[j])
            result.append(chunk[best_idx])

        return result

    # ------------------------------------------------------------------
    # Role consolidation
    # ------------------------------------------------------------------

    @staticmethod
    def _consolidate_roles(messages):
        """Merge consecutive same-role messages, preserving tool integrity.

        After silent eviction, two consecutive messages may share the same role.
        This concatenates their content with \\n\\n — EXCEPT when either message
        contains tool_calls, tool_use, tool_result, or has role 'tool'.
        """
        if not messages:
            return messages

        result = [messages[0]]
        for msg in messages[1:]:
            prev = result[-1]
            if msg.get("role") == prev.get("role") and not _is_tool_message(msg) and not _is_tool_message(prev):
                result[-1] = _merge_content(prev, msg)
            else:
                result.append(msg)
        return result

    # ------------------------------------------------------------------
    # Truncation helpers
    # ------------------------------------------------------------------

    def _truncate_messages(self, messages, provider, model):
        """Truncate individual non-system messages."""
        max_tokens = self.profile["max_msg_tokens"]
        result = []
        for m in messages:
            if m.get("role") == "system":
                result.append(m)
                continue
            tokens = self._count_tokens([m], provider, model)
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
        """Truncate system messages that exceed the budget."""
        result = []
        for m in system_msgs:
            if self._count_tokens([m], provider, model) > max_tokens:
                m = dict(m)
                m["content"] = self._do_truncate(m.get("content", ""), max_tokens)
            result.append(m)
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


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _merge_content(msg_a, msg_b):
    """Merge two same-role messages into one, concatenating content."""
    merged = dict(msg_a)
    a = msg_a.get("content", "")
    b = msg_b.get("content", "")

    if isinstance(a, list) and isinstance(b, list):
        merged["content"] = a + b
    elif isinstance(a, list):
        merged["content"] = a + [{"type": "text", "text": str(b)}]
    elif isinstance(b, list):
        merged["content"] = [{"type": "text", "text": str(a)}] + b
    else:
        merged["content"] = str(a) + "\n\n" + str(b)
    return merged
