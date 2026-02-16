"""
Bench Inference: Head pruning and KV cache compression (inlined for standalone package).

Provides head_importance_scores, find_prunable_heads, analyze_kv_cache,
compress_kv_cache, StreamingKVCache, and estimate_savings.

Original: aip.inference by Carmen Esteban
"""

import numpy as np
from scipy import sparse


# --- Attention head analysis ---

def head_importance_scores(attn, method="l1"):
    """Rank attention heads by importance.

    Parameters
    ----------
    attn : numpy.ndarray
        Attention weights (num_heads, seq_len, seq_len)
        or (batch, num_heads, seq_len, seq_len).
    method : str
        Scoring method: 'l1', 'entropy', 'max', 'variance'.

    Returns
    -------
    numpy.ndarray
        Importance scores, shape (num_heads,).
    """
    attn = np.asarray(attn, dtype=np.float64)

    if attn.ndim == 4:
        attn = np.mean(attn, axis=0)
    if attn.ndim != 3:
        raise ValueError(
            f"Expected 3D or 4D attention array, got shape {attn.shape}"
        )

    num_heads = attn.shape[0]
    scores = np.zeros(num_heads)

    for h in range(num_heads):
        head = attn[h]
        if method == "l1":
            scores[h] = np.sum(np.abs(head))
        elif method == "entropy":
            safe = np.clip(head, 1e-12, None)
            ent = -np.sum(safe * np.log(safe), axis=-1)
            scores[h] = -np.mean(ent)
        elif method == "max":
            scores[h] = np.max(head)
        elif method == "variance":
            scores[h] = np.var(head)
        else:
            raise ValueError(f"Unknown method: {method}")

    return scores


def find_prunable_heads(attn, prune_ratio=0.25):
    """Find heads that can be pruned based on importance.

    Parameters
    ----------
    attn : numpy.ndarray
        Attention weights (num_heads, seq_len, seq_len)
        or (batch, num_heads, seq_len, seq_len).
    prune_ratio : float
        Fraction of heads to mark for pruning (0-1).

    Returns
    -------
    dict
        mask, prunable_indices, keep_indices, scores.
    """
    scores = head_importance_scores(attn, method="l1")
    num_heads = len(scores)
    num_to_prune = max(1, int(num_heads * prune_ratio))
    num_to_prune = min(num_to_prune, num_heads - 1)

    sorted_indices = np.argsort(scores)
    prunable = sorted_indices[:num_to_prune]

    mask = np.ones(num_heads, dtype=bool)
    mask[prunable] = False

    return {
        "mask": mask,
        "prunable_indices": sorted(prunable.tolist()),
        "keep_indices": sorted(np.where(mask)[0].tolist()),
        "scores": scores,
    }


# --- KV cache ---

def _position_importance(keys, values):
    """Score each position by importance (L2 norm of key + value)."""
    seq_len = keys.shape[-2]
    k_flat = keys.reshape(-1, seq_len, keys.shape[-1])
    v_flat = values.reshape(-1, seq_len, values.shape[-1])

    k_norm = np.mean(np.linalg.norm(k_flat, axis=-1), axis=0)
    v_norm = np.mean(np.linalg.norm(v_flat, axis=-1), axis=0)

    return k_norm + v_norm


def analyze_kv_cache(keys, values, recent_window=64):
    """Analyze KV cache and recommend compression strategy.

    Parameters
    ----------
    keys : numpy.ndarray
        Key cache, shape (..., seq_len, head_dim).
    values : numpy.ndarray
        Value cache, shape (..., seq_len, head_dim).
    recent_window : int
        Number of recent positions to always keep.

    Returns
    -------
    dict
        Analysis with recommended strategy, estimated savings, etc.
    """
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    seq_len = keys.shape[-2]
    head_dim = keys.shape[-1]

    importance = _position_importance(keys, values)

    k_zeros = np.mean(np.abs(keys) < 1e-6)
    v_zeros = np.mean(np.abs(values) < 1e-6)

    current_bytes = keys.nbytes + values.nbytes

    if k_zeros > 0.5 or v_zeros > 0.5:
        strategy = "sparse"
        reason = f"High sparsity (keys={k_zeros:.1%}, values={v_zeros:.1%})"
        estimated_ratio = max(1.0 - max(k_zeros, v_zeros) * 0.8, 0.2)
    elif seq_len > recent_window * 4:
        strategy = "evict"
        reason = f"Long sequence ({seq_len} >> {recent_window} window)"
        estimated_ratio = (recent_window + seq_len * 0.3) / seq_len
    else:
        strategy = "merge"
        reason = f"Moderate sequence length ({seq_len})"
        estimated_ratio = 0.5

    return {
        "seq_len": seq_len,
        "head_dim": head_dim,
        "current_bytes": current_bytes,
        "estimated_compressed_bytes": int(current_bytes * estimated_ratio),
        "compression_ratio": round(1.0 / max(estimated_ratio, 0.01), 2),
        "strategy": strategy,
        "reason": reason,
        "key_sparsity": round(float(k_zeros), 4),
        "value_sparsity": round(float(v_zeros), 4),
        "recent_window": recent_window,
    }


def compress_kv_cache(keys, values, method="evict", recent_window=64,
                      chunk_size=16, keep_ratio=0.5):
    """Compress KV cache using specified method.

    Parameters
    ----------
    keys : numpy.ndarray
        Key cache, shape (..., seq_len, head_dim).
    values : numpy.ndarray
        Value cache, shape (..., seq_len, head_dim).
    method : str
        Compression method: 'evict', 'merge', or 'sparse'.
    recent_window : int
        Number of recent positions to always preserve.
    chunk_size : int
        Chunk size for merge method.
    keep_ratio : float
        Fraction of non-recent positions to keep (evict method).

    Returns
    -------
    dict
        keys, values, original_len, compressed_len, method, positions_kept.
    """
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    seq_len = keys.shape[-2]

    if method == "evict":
        return _compress_evict(keys, values, seq_len, recent_window, keep_ratio)
    elif method == "merge":
        return _compress_merge(keys, values, seq_len, recent_window, chunk_size)
    elif method == "sparse":
        return _compress_sparse(keys, values)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'evict', 'merge', or 'sparse'.")


def _compress_evict(keys, values, seq_len, recent_window, keep_ratio):
    if seq_len <= recent_window:
        return {
            "keys": keys, "values": values,
            "original_len": seq_len, "compressed_len": seq_len,
            "method": "evict", "positions_kept": np.arange(seq_len),
        }

    importance = _position_importance(keys, values)
    old_len = seq_len - recent_window
    num_keep_old = max(1, int(old_len * keep_ratio))

    old_importance = importance[:old_len]
    top_old = np.argsort(old_importance)[-num_keep_old:]
    top_old_sorted = np.sort(top_old)

    recent_positions = np.arange(old_len, seq_len)
    positions_kept = np.concatenate([top_old_sorted, recent_positions])

    return {
        "keys": keys[..., positions_kept, :],
        "values": values[..., positions_kept, :],
        "original_len": seq_len,
        "compressed_len": len(positions_kept),
        "method": "evict",
        "positions_kept": positions_kept,
    }


def _compress_merge(keys, values, seq_len, recent_window, chunk_size):
    if seq_len <= recent_window:
        return {
            "keys": keys, "values": values,
            "original_len": seq_len, "compressed_len": seq_len,
            "method": "merge", "positions_kept": np.arange(seq_len),
        }

    old_len = seq_len - recent_window
    merged_keys_list = []
    merged_values_list = []
    chunk_positions = []

    for start in range(0, old_len, chunk_size):
        end = min(start + chunk_size, old_len)
        chunk_k = np.mean(keys[..., start:end, :], axis=-2, keepdims=True)
        chunk_v = np.mean(values[..., start:end, :], axis=-2, keepdims=True)
        merged_keys_list.append(chunk_k)
        merged_values_list.append(chunk_v)
        chunk_positions.append(start)

    recent_k = keys[..., old_len:, :]
    recent_v = values[..., old_len:, :]

    new_keys = np.concatenate(merged_keys_list + [recent_k], axis=-2)
    new_values = np.concatenate(merged_values_list + [recent_v], axis=-2)

    positions_kept = np.concatenate([
        np.array(chunk_positions),
        np.arange(old_len, seq_len),
    ])

    return {
        "keys": new_keys, "values": new_values,
        "original_len": seq_len, "compressed_len": new_keys.shape[-2],
        "method": "merge", "positions_kept": positions_kept,
    }


def _compress_sparse(keys, values):
    seq_len = keys.shape[-2]
    k_2d = keys.reshape(-1, keys.shape[-1])
    v_2d = values.reshape(-1, values.shape[-1])

    k_sparse = sparse.csr_matrix(k_2d)
    v_sparse = sparse.csr_matrix(v_2d)

    return {
        "keys": k_sparse, "values": v_sparse,
        "original_len": seq_len, "compressed_len": seq_len,
        "method": "sparse",
        "key_nnz": k_sparse.nnz, "value_nnz": v_sparse.nnz,
        "original_shape": keys.shape,
    }


class StreamingKVCache:
    """Streaming KV cache manager with automatic compression."""

    def __init__(self, head_dim, max_len=512, recent_window=64,
                 compress_method="evict"):
        self.head_dim = head_dim
        self.max_len = max_len
        self.recent_window = recent_window
        self.compress_method = compress_method
        self._keys = np.empty((0, head_dim), dtype=np.float64)
        self._values = np.empty((0, head_dim), dtype=np.float64)
        self._num_compressions = 0

    def add_tokens(self, new_keys, new_values):
        new_keys = np.asarray(new_keys, dtype=np.float64)
        new_values = np.asarray(new_values, dtype=np.float64)
        if new_keys.ndim == 1:
            new_keys = new_keys.reshape(1, -1)
            new_values = new_values.reshape(1, -1)
        self._keys = np.concatenate([self._keys, new_keys], axis=0)
        self._values = np.concatenate([self._values, new_values], axis=0)
        if self._keys.shape[0] > self.max_len:
            self._compress()

    def _compress(self):
        result = compress_kv_cache(
            self._keys, self._values,
            method=self.compress_method,
            recent_window=self.recent_window,
        )
        self._keys = np.asarray(result["keys"], dtype=np.float64)
        self._values = np.asarray(result["values"], dtype=np.float64)
        self._num_compressions += 1

    @property
    def current_len(self):
        return self._keys.shape[0]

    @property
    def keys(self):
        return self._keys

    @property
    def values(self):
        return self._values

    @property
    def num_compressions(self):
        return self._num_compressions

    def __repr__(self):
        return (
            f"StreamingKVCache(len={self.current_len}, "
            f"head_dim={self.head_dim}, max_len={self.max_len}, "
            f"compressions={self._num_compressions})"
        )


def estimate_savings(attn=None, keys=None, values=None,
                     prune_ratio=0.25, compress_method="evict",
                     recent_window=64):
    """Estimate total inference savings from head pruning + KV compression.

    Parameters
    ----------
    attn : numpy.ndarray, optional
        Attention weights (num_heads, seq_len, seq_len).
    keys : numpy.ndarray, optional
        Key cache (..., seq_len, head_dim).
    values : numpy.ndarray, optional
        Value cache (..., seq_len, head_dim).
    prune_ratio : float
        Fraction of heads to prune.
    compress_method : str
        KV compression method.
    recent_window : int
        Recent window size for KV compression.

    Returns
    -------
    dict
        Summary of estimated savings from both techniques.
    """
    result = {
        "head_pruning": None,
        "kv_compression": None,
        "total_param_reduction_ratio": 0.0,
        "total_memory_reduction_ratio": 0.0,
    }

    if attn is not None:
        attn = np.asarray(attn, dtype=np.float64)
        prune_result = find_prunable_heads(attn, prune_ratio=prune_ratio)
        num_heads = len(prune_result["scores"])
        num_kept = len(prune_result["keep_indices"])
        head_ratio = 1.0 - (num_kept / max(num_heads, 1))

        result["head_pruning"] = {
            "num_heads": num_heads,
            "num_pruned": num_heads - num_kept,
            "num_kept": num_kept,
            "param_reduction_ratio": round(head_ratio, 4),
        }
        result["total_param_reduction_ratio"] = head_ratio

    if keys is not None and values is not None:
        keys = np.asarray(keys, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        analysis = analyze_kv_cache(keys, values, recent_window=recent_window)
        original_bytes = analysis["current_bytes"]
        estimated_bytes = analysis["estimated_compressed_bytes"]
        mem_ratio = 1.0 - (estimated_bytes / max(original_bytes, 1))

        result["kv_compression"] = {
            "strategy": analysis["strategy"],
            "original_bytes": original_bytes,
            "estimated_compressed_bytes": estimated_bytes,
            "memory_reduction_ratio": round(mem_ratio, 4),
            "compression_ratio": analysis["compression_ratio"],
        }
        result["total_memory_reduction_ratio"] = mem_ratio

    return result
