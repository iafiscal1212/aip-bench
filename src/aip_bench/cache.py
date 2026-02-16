"""
Bench Cache: Disk-based caching for model outputs.

Avoids re-running expensive model inference by caching
(model_name, prompt) -> result on disk as JSON files.

Usage:
    from aip_bench.cache import ResultCache

    cache = ResultCache()
    result = cache.get("distilgpt2", "What is 2+2?")
    if result is None:
        result = model.generate("What is 2+2?")
        cache.put("distilgpt2", "What is 2+2?", result)

Author: Carmen Esteban
"""

import hashlib
import json
from pathlib import Path


class ResultCache:
    """Disk-based cache for model outputs.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for cached files. Default: ~/.cache/aip-bench
    enabled : bool
        Whether caching is active.
    """

    def __init__(self, cache_dir=None, enabled=True):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "aip-bench"
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model_name, prompt):
        """Generate cache key from model name and prompt."""
        raw = f"{model_name}\x00{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model_name, prompt):
        """Get cached result or None.

        Parameters
        ----------
        model_name : str
            Model identifier.
        prompt : str
            Input prompt.

        Returns
        -------
        object or None
            Cached result, or None if not cached.
        """
        if not self.enabled:
            return None
        path = self.cache_dir / f"{self._key(model_name, prompt)}.json"
        if path.exists():
            self._hits += 1
            with open(path) as f:
                return json.load(f)
        self._misses += 1
        return None

    def put(self, model_name, prompt, result):
        """Cache a result.

        Parameters
        ----------
        model_name : str
            Model identifier.
        prompt : str
            Input prompt.
        result : object
            JSON-serializable result.
        """
        if not self.enabled:
            return
        path = self.cache_dir / f"{self._key(model_name, prompt)}.json"
        with open(path, "w") as f:
            json.dump(result, f)

    def clear(self):
        """Delete all cached results."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self):
        """Cache hit/miss statistics.

        Returns
        -------
        dict
            {hits, misses, total, hit_rate}
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / max(total, 1),
        }

    def __repr__(self):
        return (
            f"ResultCache(dir={self.cache_dir}, "
            f"enabled={self.enabled}, stats={self.stats})"
        )
