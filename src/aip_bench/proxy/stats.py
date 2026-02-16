"""
Compression statistics tracking for the proxy.

Records per-request and cumulative token savings, compression ratios,
and session-level summaries.
"""

import time


class CompressionStats:
    """Track compression metrics across proxy requests."""

    def __init__(self):
        self.requests = 0
        self.compressions = 0
        self.tokens_before_total = 0
        self.tokens_after_total = 0
        self._last = {}
        self._start_time = time.monotonic()

    def record(self, tokens_before, tokens_after):
        """Record a single compression event."""
        self.requests += 1
        self.compressions += 1
        self.tokens_before_total += tokens_before
        self.tokens_after_total += tokens_after
        self._last = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_before - tokens_after,
            "ratio": tokens_after / tokens_before if tokens_before > 0 else 1.0,
        }

    def record_passthrough(self):
        """Record a request that was not compressed."""
        self.requests += 1

    def last(self):
        """Return stats from the most recent compression."""
        return dict(self._last)

    def summary(self):
        """Return cumulative session statistics."""
        saved = self.tokens_before_total - self.tokens_after_total
        elapsed = time.monotonic() - self._start_time
        return {
            "requests": self.requests,
            "compressions": self.compressions,
            "tokens_before_total": self.tokens_before_total,
            "tokens_after_total": self.tokens_after_total,
            "tokens_saved": saved,
            "savings_pct": (
                (saved / self.tokens_before_total * 100)
                if self.tokens_before_total > 0
                else 0.0
            ),
            "avg_compression_ratio": (
                (self.tokens_after_total / self.tokens_before_total)
                if self.tokens_before_total > 0
                else 1.0
            ),
            "elapsed_seconds": round(elapsed, 2),
        }
