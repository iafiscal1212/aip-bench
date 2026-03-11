"""
Tests for aip_bench.proxy — Accordion LLM proxy.

All tests use pure logic only (no network, no aiohttp runtime).
"""

import pytest

from aip_bench.proxy.accordion import (
    MessageAccordion,
    estimate_tokens,
    PROFILES,
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
        # ~11 chars / 4 = 2 tokens, minimum 1
        assert tokens >= 1

    def test_empty_messages(self):
        # No messages → minimum 1
        assert estimate_tokens([]) == 1

    def test_long_content(self):
        msgs = [{"role": "user", "content": "a" * 4000}]
        tokens = estimate_tokens(msgs)
        # 4000 / 4 = 1000 tokens
        assert tokens == 1000

    def test_content_blocks(self):
        """Anthropic-style content blocks."""
        msgs = [{"role": "user", "content": [{"text": "Hello " * 100}]}]
        tokens = estimate_tokens(msgs)
        assert tokens > 20


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
        # Token count should decrease (atoms silently evicted, no markers)
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
    def test_merge_reduces_messages(self):
        """Aggressive profile uses merge, silently dropping low-value atoms."""
        accordion = MessageAccordion(profile="aggressive")
        # aggressive recent_window = 6
        msgs = _make_conversation(n_old=30, n_recent=6)
        compressed, stats = accordion.compress(msgs, model="gpt-4")
        assert stats["compressed"] is True
        # Silent eviction: fewer messages, no injected markers
        assert stats["messages_after"] < stats["messages_before"]
        assert stats["tokens_after"] < stats["tokens_before"]
        # No CONTEXT markers should be present
        for m in compressed:
            content = m.get("content", "")
            if isinstance(content, str):
                assert "[CONTEXT:" not in content


# ---------------------------------------------------------------------------
# Threshold / no-compress
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_no_compress_below_threshold(self):
        """Should not compress when token count is below min_tokens."""
        accordion = MessageAccordion(profile="conservative")
        # conservative min_tokens = 8000
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
        """Balanced profile should compress conversations over 5K tokens."""
        accordion = MessageAccordion(profile="balanced")
        # balanced min_tokens = 5000 -> need enough tokens
        msgs = [{"role": "system", "content": "System prompt."}]
        # Need distinct text or much more of it for tiktoken to hit 5k
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            # More varied text so it doesn't compress as well in BPE
            msgs.append({"role": role, "content": f"Message {i}: " + f"word {i} " * 150})
        compressed, stats = accordion.compress(msgs)
        assert stats["compressed"] is True
        savings_pct = stats["tokens_saved"] / stats["tokens_before"] * 100
        assert savings_pct > 10  # At least 10% savings

    def test_absolute_threshold_fires_early(self):
        """min_tokens triggers on modest conversations, not just near context limit."""
        accordion = MessageAccordion(profile="balanced")
        # ~6K tokens needed. Let's use more unique data.
        msgs = [{"role": "system", "content": "Be helpful."}]
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Msg {i}: " + f"Unique data {i} " * 100})
        compressed, stats = accordion.compress(msgs)
        assert stats["compressed"] is True


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


# ---------------------------------------------------------------------------
# Content block truncation (Anthropic tool_result)
# ---------------------------------------------------------------------------

class TestContentBlockTruncation:
    def test_truncates_large_tool_result_blocks(self):
        """Anthropic tool_result with large file content gets truncated."""
        accordion = MessageAccordion(profile="aggressive")
        big_file = "line\n" * 5000  # ~25K chars -> ~6K tokens
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [
                {"type": "tool_result", "text": big_file},
            ]},
        ]
        # Add enough messages to trigger compression
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"msg {i}: " + "y" * 400})

        compressed, stats = accordion.compress(msgs)
        # The tool_result block should have been truncated
        tool_msg = compressed[1]  # after system
        if isinstance(tool_msg["content"], list):
            text = tool_msg["content"][0].get("text", "")
        else:
            text = str(tool_msg["content"])
        assert len(text) < len(big_file)
        assert "[...truncated...]" in text

    def test_small_blocks_untouched(self):
        """Small content blocks pass through unchanged."""
        accordion = MessageAccordion(profile="balanced")
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        compressed, stats = accordion.compress(msgs)
        assert compressed[0]["content"][0]["text"] == "Hello"


# ---------------------------------------------------------------------------
# System prompt compression
# ---------------------------------------------------------------------------

class TestSystemCompression:
    def test_large_system_prompt_compressed(self):
        """balanced/aggressive profiles truncate large system prompts."""
        accordion = MessageAccordion(profile="balanced")
        # balanced max_system_tokens = 4096 -> 16K chars
        huge_system = "You are helpful. " * 2000  # ~34K chars -> ~8.5K tokens
        msgs = [
            {"role": "system", "content": huge_system},
        ]
        # Need enough total tokens to pass min_tokens threshold
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Msg {i}: " + "z" * 800})

        compressed, stats = accordion.compress(msgs)
        sys_msg = [m for m in compressed if m["role"] == "system"][0]
        # System content should be shorter than original
        assert len(str(sys_msg["content"])) < len(huge_system)
        assert "[...truncated...]" in str(sys_msg["content"])

    def test_conservative_no_system_compression(self):
        """Conservative profile (max_system_tokens=0) never compresses system."""
        accordion = MessageAccordion(profile="conservative")
        huge_system = "You are helpful. " * 2000
        msgs = [{"role": "system", "content": huge_system}]
        for i in range(60):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Msg {i}: " + "z" * 800})
        compressed, stats = accordion.compress(msgs)
        sys_msg = [m for m in compressed if m["role"] == "system"][0]
        assert sys_msg["content"] == huge_system


# ---------------------------------------------------------------------------
# Merge preserves high-score messages
# ---------------------------------------------------------------------------

class TestMergeKeepsBest:
    def test_merge_keeps_highest_scoring(self):
        """Merge keeps the best message per chunk, not just first/last."""
        accordion = MessageAccordion(profile="aggressive")
        msgs = [{"role": "system", "content": "System."}]
        # Add old messages with one code-heavy message in the middle
        for i in range(20):
            if i == 10:
                # This one has code + error -> high score
                msgs.append({"role": "user", "content": "```python\nraise Error\n```\nerror traceback"})
            else:
                msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": "filler " * 50})
        # Recent window
        for i in range(6):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": "recent " * 50})

        compressed, stats = accordion.compress(msgs, model="gpt-4")
        # The code+error message should be preserved verbatim
        preserved = [m for m in compressed if "```python" in m.get("content", "") and "error traceback" in m.get("content", "")]
        assert len(preserved) == 1


# ---------------------------------------------------------------------------
# Stdlib fallback server import
# ---------------------------------------------------------------------------

class TestStdlibFallback:
    def test_stdlib_server_importable(self):
        """StdlibProxyServer can be imported without aiohttp."""
        from aip_bench.proxy.server_stdlib import StdlibProxyServer
        server = StdlibProxyServer(port=9999, profile="balanced", target="http://localhost:1234")
        assert server.port == 9999
        assert server.accordion.profile_name == "balanced"


# ---------------------------------------------------------------------------
# Anthropic provider tool use validation
# ---------------------------------------------------------------------------

class TestAnthropicProviderValidation:
    def setup_method(self):
        self.provider = AnthropicProvider()

    def test_validate_tool_use_removes_invalid_tool_result(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "70 degrees and sunny",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "invalid_id",
                        "content": "some other result",
                    },
                ],
            },
        ]

        validated_messages = self.provider._validate_tool_use(messages)

        # The user message should be at index 1
        user_message_content = validated_messages[1].get("content", [])

        # Check that the invalid tool_result was removed
        assert len(user_message_content) == 1
        assert user_message_content[0].get("tool_use_id") == "tool_123"

    def test_validate_tool_use_keeps_valid_tool_results(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    },
                    {"type": "tool_use", "id": "tool_456", "name": "get_time", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "70 degrees and sunny",
                    },
                    {"type": "tool_result", "tool_use_id": "tool_456", "content": "10:00 AM"},
                ],
            },
        ]

        validated_messages = self.provider._validate_tool_use(messages)
        user_message_content = validated_messages[1].get("content", [])

        assert len(user_message_content) == 2

    def test_validate_tool_use_no_tool_results(self):
        messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ]

        validated_messages = self.provider._validate_tool_use(messages)
        assert messages == validated_messages

    def test_validate_tool_use_no_tool_use(self):
        messages = [
            {"role": "assistant", "content": "Hello"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "70 degrees and sunny",
                    }
                ],
            },
        ]

        validated_messages = self.provider._validate_tool_use(messages)
        user_message_content = validated_messages[1].get("content", [])

        # All tool_results should be removed
        assert len(user_message_content) == 0

    def test_filter_empty_user_messages(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": None},  # Should be removed
            {"role": "user", "content": ""},    # Should be removed
            {"role": "user", "content": "   "}, # Should be removed (whitespace only)
            {"role": "user", "content": []},    # Should be removed
            {"role": "user", "content": ["Hello"]}, # Should be kept
            {"role": "assistant", "content": "Ok"},
        ]
        
        filtered_messages = self.provider._filter_empty_user_messages(messages)
        
        expected_messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": ["Hello"]},
            {"role": "assistant", "content": "Ok"},
        ]
        
        assert filtered_messages == expected_messages


# ---------------------------------------------------------------------------
# AnthropicProvider — new compatibility fixes
# ---------------------------------------------------------------------------

class TestAnthropicProviderFixes:
    def setup_method(self):
        self.provider = AnthropicProvider()

    # --- Bug 1: validation order ---

    def test_validation_order_empties_are_cleaned_after_tool_validation(self):
        """After _validate_tool_use removes tool_results, the now-empty user
        message must also be removed. Previously filter_empty ran first so it
        was a no-op at that point, leaving an empty user message.
        The leading assistant message causes a placeholder user turn to be
        injected, so the final messages start with the injected user turn."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    # placeholder injected by accordion — no tool_use IDs
                    {"type": "text", "text": "[1 previous message omitted]"},
                ],
            },
            {
                "role": "user",
                "content": [
                    # This tool_result has no matching tool_use in the preceding msg
                    {"type": "tool_result", "tool_use_id": "dangling_id", "content": "data"},
                ],
            },
        ]
        body = {"messages": messages}
        result = self.provider.replace_messages(body, messages)
        msgs = result["messages"]
        # The dangling tool_result user message was removed (now empty → filtered)
        # but a placeholder user turn was injected because messages started with assistant
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "[Context omitted]"
        # No user message with tool_result content remains
        for m in msgs:
            content = m.get("content", [])
            if isinstance(content, list):
                assert not any(b.get("type") == "tool_result" for b in content)

    # --- Bug 2: consecutive same-role messages ---

    def test_merge_consecutive_assistant_messages_string(self):
        messages = [
            {"role": "assistant", "content": "First part."},
            {"role": "assistant", "content": "Second part."},
        ]
        merged = self.provider._merge_consecutive_roles(messages)
        assert len(merged) == 1
        assert "First part." in merged[0]["content"]
        assert "Second part." in merged[0]["content"]

    def test_merge_consecutive_user_messages_list(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "A"}]},
            {"role": "user", "content": [{"type": "text", "text": "B"}]},
        ]
        merged = self.provider._merge_consecutive_roles(messages)
        assert len(merged) == 1
        assert len(merged[0]["content"]) == 2

    def test_merge_consecutive_mixed_content(self):
        """String + list should produce a list with a text block prepended."""
        messages = [
            {"role": "assistant", "content": "Summary text"},
            {"role": "assistant", "content": [{"type": "text", "text": "Kept content"}]},
        ]
        merged = self.provider._merge_consecutive_roles(messages)
        assert len(merged) == 1
        content = merged[0]["content"]
        assert isinstance(content, list)
        assert any(b.get("text") == "Summary text" for b in content)

    def test_no_merge_for_alternating_roles(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        merged = self.provider._merge_consecutive_roles(messages)
        assert len(merged) == 3

    # --- Bug 3: system-role messages in messages array ---

    def test_normalize_system_prompt_extracts_to_top_level(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        body = {"messages": messages}
        result = self.provider.replace_messages(body, messages)
        assert result["system"] == "You are a helpful assistant."
        assert all(m["role"] != "system" for m in result["messages"])

    def test_normalize_system_prompt_merges_with_existing_system(self):
        messages = [
            {"role": "system", "content": "Extra system context."},
            {"role": "user", "content": "Hello"},
        ]
        body = {"messages": messages, "system": "Original system prompt."}
        result = self.provider.replace_messages(body, messages)
        assert "Extra system context." in result["system"]
        assert "Original system prompt." in result["system"]

    def test_normalize_system_prompt_no_system_messages(self):
        """If no system-role messages, body['system'] is untouched."""
        messages = [{"role": "user", "content": "Hi"}]
        body = {"messages": messages, "system": "Keep me."}
        result = self.provider.replace_messages(body, messages)
        assert result["system"] == "Keep me."

    # --- Bug 4: ensure starts with user ---

    def test_ensure_starts_with_user_injects_placeholder(self):
        """Leading assistant messages are preserved; a placeholder user turn is
        injected instead of dropping context."""
        messages = [
            {"role": "assistant", "content": "[1 message omitted]"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.provider._ensure_starts_with_user(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Context omitted]"
        assert len(result) == 4  # placeholder + original 3

    def test_ensure_starts_with_user_already_correct(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.provider._ensure_starts_with_user(messages)
        assert result == messages

    def test_ensure_starts_with_user_empty(self):
        assert self.provider._ensure_starts_with_user([]) == []

    # --- replace_messages end-to-end ---

    def test_merge_before_validate_preserves_valid_tool_result(self):
        """Regression: when accordion produces consecutive assistant messages
        (tool_use block + summary placeholder), the tool_result in the following
        user message must NOT be stripped as orphaned.
        Merge must happen before validate_tool_use."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "fn", "input": {}},
                ],
            },
            # accordion inserted a summary placeholder → consecutive assistant
            {"role": "assistant", "content": "[1 previous message omitted]"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                ],
            },
        ]
        body = {"messages": messages}
        result = self.provider.replace_messages(body, messages)
        msgs = result["messages"]
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "[Context omitted]"
        content = user_msgs[1]["content"]
        assert isinstance(content, list) and len(content) == 1
        assert content[0]["tool_use_id"] == "t1"

    def test_replace_messages_full_pipeline(self):
        """Full pipeline: system extracted, orphaned tool_result removed,
        consecutive roles merged, starts with user."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "assistant", "content": "[1 omitted]"},  # leading non-user
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": " there"},        # consecutive assistant
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "gone_id", "content": "result"},
            ]},
        ]
        body = {"messages": messages}
        result = self.provider.replace_messages(body, messages)

        msgs = result["messages"]
        # system extracted
        assert result.get("system") == "Be helpful."
        assert all(m["role"] != "system" for m in msgs)
        # starts with user
        assert msgs[0]["role"] == "user"
        # no orphaned tool_results (user msg was emptied and removed)
        for m in msgs:
            content = m.get("content", [])
            if isinstance(content, list):
                assert not any(b.get("type") == "tool_result" for b in content)
