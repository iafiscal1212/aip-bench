"""
Tests for aip_bench.proxy — Accordion LLM proxy.

All tests use pure logic only (no network, no aiohttp runtime).
"""

import pytest

from aip_bench.proxy.accordion import (
    MessageAccordion,
    estimate_tokens,
    PROFILES,
    MODEL_CONTEXTS,
    ROLE_WEIGHTS,
)
from aip_bench.proxy.providers import (
    detect_provider,
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    GenericProvider,
)
from aip_bench.proxy.stats import CompressionStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n, role="user", content_len=100):
    """Generate n messages with given role and approximate content length."""
    return [
        {"role": role, "content": "x" * content_len}
        for _ in range(n)
    ]


def _make_conversation(n_old, n_recent=4, content_len=800):
    """Build a conversation with system + old + recent messages."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_old):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Old message {i}: " + "a" * content_len})
    for i in range(n_recent):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Recent message {i}: " + "b" * content_len})
    return msgs


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        tokens = estimate_tokens(msgs)
        assert tokens >= 1
        # "Hello world" = 11 chars -> ~2-3 tokens
        assert tokens == 11 // 4 or tokens == max(11 // 4, 1)

    def test_empty_messages(self):
        assert estimate_tokens([]) == 1  # min 1

    def test_long_content(self):
        msgs = [{"role": "user", "content": "a" * 4000}]
        tokens = estimate_tokens(msgs)
        assert tokens == 1000

    def test_content_blocks(self):
        """Anthropic-style content blocks."""
        msgs = [{"role": "user", "content": [{"text": "Hello " * 100}]}]
        tokens = estimate_tokens(msgs)
        assert tokens > 10


# ---------------------------------------------------------------------------
# MessageAccordion._score
# ---------------------------------------------------------------------------

class TestScoreMessage:
    def setup_method(self):
        self.accordion = MessageAccordion(profile="balanced")

    def test_code_block_scores_high(self):
        msg = {"role": "user", "content": "Here is code:\n```python\nprint('hi')\n```"}
        score = self.accordion._score(msg)
        plain = {"role": "user", "content": "Here is some text without code"}
        score_plain = self.accordion._score(plain)
        assert score > score_plain

    def test_error_message_scores_high(self):
        msg = {"role": "assistant", "content": "Error: something failed with traceback"}
        score = self.accordion._score(msg)
        plain = {"role": "assistant", "content": "Everything works fine"}
        score_plain = self.accordion._score(plain)
        assert score > score_plain

    def test_short_message_scores_low(self):
        short = {"role": "assistant", "content": "ok"}
        long = {"role": "assistant", "content": "a" * 2000}
        assert self.accordion._score(short) < self.accordion._score(long)

    def test_system_message_scores_highest(self):
        sys_msg = {"role": "system", "content": "You are helpful."}
        user_msg = {"role": "user", "content": "You are helpful."}
        assert self.accordion._score(sys_msg) > self.accordion._score(user_msg)

    def test_score_capped_at_one(self):
        msg = {
            "role": "system",
            "content": "```error traceback failed``` " + "a" * 10000,
        }
        assert self.accordion._score(msg) <= 1.0


# ---------------------------------------------------------------------------
# Evict strategy
# ---------------------------------------------------------------------------

class TestEvict:
    def test_keeps_recent_window(self):
        """Recent window messages are never touched by compression."""
        accordion = MessageAccordion(profile="balanced")
        # balanced recent_window = 12
        msgs = _make_conversation(n_old=40, n_recent=12)
        # Force compression by using a tiny model context
        compressed, stats = accordion.compress(msgs, model="gpt-4")
        # The last 12 non-system messages should be untouched
        recent_original = msgs[-12:]
        recent_compressed = compressed[-12:]
        assert recent_original == recent_compressed

    def test_removes_low_score_messages(self):
        """Evict removes messages, reducing token count."""
        accordion = MessageAccordion(profile="balanced")
        msgs = _make_conversation(n_old=40, n_recent=12)
        compressed, stats = accordion.compress(msgs, model="gpt-4")
        assert stats["compressed"] is True
        # Token count should decrease even if message count stays similar
        # (evicted messages replaced by short summary placeholders)
        assert stats["tokens_after"] < stats["tokens_before"]

    def test_preserves_system_prompt(self):
        """System messages are never removed."""
        accordion = MessageAccordion(profile="balanced")
        msgs = _make_conversation(n_old=40, n_recent=12)
        compressed, stats = accordion.compress(msgs, model="gpt-4")
        system_msgs = [m for m in compressed if m["role"] == "system"]
        assert len(system_msgs) >= 1
        assert system_msgs[0]["content"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Merge strategy
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_chunks_old_messages(self):
        """Aggressive profile uses merge, producing summary messages."""
        accordion = MessageAccordion(profile="aggressive")
        # aggressive recent_window = 6
        msgs = _make_conversation(n_old=30, n_recent=6)
        compressed, stats = accordion.compress(msgs, model="gpt-4")
        assert stats["compressed"] is True
        # Merged messages contain the marker
        merged = [m for m in compressed if "[Merged" in m.get("content", "")]
        assert len(merged) >= 1


# ---------------------------------------------------------------------------
# Threshold / no-compress
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_no_compress_below_threshold(self):
        """Should not compress when token count is below threshold."""
        accordion = MessageAccordion(profile="conservative")
        # conservative threshold = 0.7, default context = 200k
        # A few short messages won't trigger compression
        msgs = [
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        compressed, stats = accordion.compress(msgs)
        assert stats["compressed"] is False
        assert compressed == msgs

    def test_compress_balanced_profile(self):
        """Balanced profile should compress large conversations."""
        accordion = MessageAccordion(profile="balanced")
        # Build a conversation that exceeds threshold for gpt-4 (8192 tokens)
        msgs = [{"role": "system", "content": "System prompt."}]
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Message {i}: " + "word " * 200})
        compressed, stats = accordion.compress(msgs, model="gpt-4")
        assert stats["compressed"] is True
        savings_pct = stats["tokens_saved"] / stats["tokens_before"] * 100
        assert savings_pct > 10  # At least 10% savings


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_detect_anthropic(self):
        provider = detect_provider("/v1/messages")
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == "anthropic"

    def test_detect_openai(self):
        provider = detect_provider("/v1/chat/completions")
        assert isinstance(provider, OpenAIProvider)
        assert provider.name == "openai"

    def test_detect_generic(self):
        provider = detect_provider("/api/generate")
        assert isinstance(provider, GenericProvider)
        assert provider.name == "generic"

    def test_anthropic_url_building(self):
        provider = AnthropicProvider()
        url = provider.build_url(None, "/v1/messages")
        assert url == "https://api.anthropic.com/v1/messages"

    def test_openai_url_building(self):
        provider = OpenAIProvider()
        url = provider.build_url(None, "/v1/chat/completions")
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_generic_no_target(self):
        provider = GenericProvider()
        url = provider.build_url(None, "/api/generate")
        assert url is None

    def test_custom_target(self):
        provider = OpenAIProvider()
        url = provider.build_url("http://localhost:11434", "/v1/chat/completions")
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_extract_replace_messages(self):
        provider = OpenAIProvider()
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        msgs = provider.extract_messages(body)
        assert len(msgs) == 1
        new_body = provider.replace_messages(body, [{"role": "user", "content": "Hello"}])
        assert new_body["messages"][0]["content"] == "Hello"
        # Original body not mutated
        assert body["messages"][0]["content"] == "Hi"

    def test_forward_headers_strips_hop_by_hop(self):
        provider = Provider()
        headers = {
            "Authorization": "Bearer sk-...",
            "Host": "api.openai.com",
            "Content-Length": "123",
            "X-Custom": "value",
        }
        forwarded = provider.forward_headers(headers)
        assert "Authorization" in forwarded
        assert "X-Custom" in forwarded
        assert "Host" not in forwarded
        assert "Content-Length" not in forwarded


# ---------------------------------------------------------------------------
# CompressionStats
# ---------------------------------------------------------------------------

class TestCompressionStats:
    def test_tracking(self):
        stats = CompressionStats()
        stats.record(1000, 600)
        assert stats.requests == 1
        assert stats.compressions == 1
        assert stats.tokens_before_total == 1000
        assert stats.tokens_after_total == 600

        last = stats.last()
        assert last["tokens_saved"] == 400
        assert abs(last["ratio"] - 0.6) < 0.01

    def test_multiple_records(self):
        stats = CompressionStats()
        stats.record(1000, 600)
        stats.record(2000, 1000)
        summary = stats.summary()
        assert summary["requests"] == 2
        assert summary["compressions"] == 2
        assert summary["tokens_saved"] == 1400
        assert abs(summary["savings_pct"] - (1400 / 3000 * 100)) < 0.1

    def test_passthrough(self):
        stats = CompressionStats()
        stats.record_passthrough()
        assert stats.requests == 1
        assert stats.compressions == 0

    def test_summary_fields(self):
        stats = CompressionStats()
        stats.record(500, 250)
        summary = stats.summary()
        assert "requests" in summary
        assert "compressions" in summary
        assert "tokens_saved" in summary
        assert "savings_pct" in summary
        assert "avg_compression_ratio" in summary
        assert "elapsed_seconds" in summary


# ---------------------------------------------------------------------------
# Accordion session (multiple requests)
# ---------------------------------------------------------------------------

class TestAccordionSession:
    def test_session_accumulates_stats(self):
        """Multiple compress() calls accumulate in shared stats."""
        accordion = MessageAccordion(profile="balanced")
        stats = CompressionStats()
        accordion.stats = stats

        # First large conversation (must exceed threshold for gpt-4)
        msgs1 = _make_conversation(n_old=40, n_recent=12, content_len=800)
        accordion.compress(msgs1, model="gpt-4")

        # Second large conversation
        msgs2 = _make_conversation(n_old=50, n_recent=12, content_len=800)
        accordion.compress(msgs2, model="gpt-4")

        summary = stats.summary()
        assert summary["compressions"] == 2
        assert summary["tokens_saved"] > 0

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            MessageAccordion(profile="nonexistent")

    def test_empty_messages(self):
        accordion = MessageAccordion(profile="balanced")
        compressed, stats = accordion.compress([])
        assert compressed == []
        assert stats["compressed"] is False
