"""
Bench Guard: Hallucination detection metrics (inlined for standalone package).

Five core attention-based metrics plus a composite hallucination_score.
All numpy-pure except cross_layer_consistency which uses scipy.stats.

Original: aip.guard.metrics by Carmen Esteban
"""

import numpy as np
from scipy import stats


def attention_entropy(attn):
    """Shannon entropy of attention weights per head.

    Parameters
    ----------
    attn : numpy.ndarray
        Attention weights with shape (..., seq_len, seq_len).

    Returns
    -------
    numpy.ndarray
        Entropy values, shape attn.shape[:-2] + (seq_len,).
    """
    attn = np.asarray(attn, dtype=np.float64)
    attn_safe = np.clip(attn, 1e-12, None)
    entropy = -np.sum(attn_safe * np.log(attn_safe), axis=-1)
    return entropy


def attention_sparsity(attn, threshold=0.01):
    """Fraction of attention weights above threshold per head.

    Parameters
    ----------
    attn : numpy.ndarray
        Attention weights with shape (..., seq_len, seq_len).
    threshold : float
        Minimum weight to count as significant.

    Returns
    -------
    numpy.ndarray
        Fraction of weights above threshold per query position.
    """
    attn = np.asarray(attn, dtype=np.float64)
    seq_len = attn.shape[-1]
    above = np.sum(attn > threshold, axis=-1)
    return above / max(seq_len, 1)


def cross_layer_consistency(layers):
    """Pearson correlation between attention patterns of adjacent layers.

    Parameters
    ----------
    layers : list of numpy.ndarray
        List of attention arrays, one per layer.

    Returns
    -------
    numpy.ndarray
        Correlation values between adjacent layers, shape (n_layers - 1,).
    """
    if len(layers) < 2:
        return np.array([1.0])

    correlations = []
    for i in range(len(layers) - 1):
        a = np.asarray(layers[i], dtype=np.float64).ravel()
        b = np.asarray(layers[i + 1], dtype=np.float64).ravel()
        if len(a) == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            correlations.append(0.0)
        else:
            r, _ = stats.pearsonr(a, b)
            correlations.append(float(r))

    return np.array(correlations)


def token_probability_stats(probs):
    """Statistics of token probabilities for generated tokens.

    Parameters
    ----------
    probs : numpy.ndarray
        Token probabilities, shape (seq_len,) or (batch, seq_len).

    Returns
    -------
    dict
        mean, variance, low_confidence_ratio.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if len(probs) == 0:
        return {"mean": 0.0, "variance": 0.0, "low_confidence_ratio": 1.0}

    return {
        "mean": float(np.mean(probs)),
        "variance": float(np.var(probs)),
        "low_confidence_ratio": float(np.mean(probs < 0.1)),
    }


def lookback_ratio(attn, context_len):
    """Ratio of attention going to context tokens vs self-generated tokens.

    Parameters
    ----------
    attn : numpy.ndarray
        Attention weights with shape (..., seq_len, seq_len).
    context_len : int
        Number of tokens that are original context.

    Returns
    -------
    float
        Ratio of attention to context tokens (0-1).
    """
    attn = np.asarray(attn, dtype=np.float64)
    seq_len = attn.shape[-1]
    if context_len <= 0 or context_len >= seq_len:
        return 1.0

    gen_attn = attn[..., context_len:, :]
    if gen_attn.size == 0:
        return 1.0

    context_weight = np.sum(gen_attn[..., :context_len])
    total_weight = np.sum(gen_attn)
    if total_weight < 1e-12:
        return 0.0

    return float(context_weight / total_weight)


def hallucination_score(
    attn=None,
    token_probs=None,
    layers=None,
    context_len=None,
    weights=None,
):
    """Composite hallucination score combining available metrics.

    Parameters
    ----------
    attn : numpy.ndarray, optional
        Attention weights (..., seq_len, seq_len).
    token_probs : numpy.ndarray, optional
        Token probabilities (seq_len,).
    layers : list of numpy.ndarray, optional
        Per-layer attention for consistency metric.
    context_len : int, optional
        Context length for lookback.
    weights : dict, optional
        Override default metric weights.

    Returns
    -------
    dict
        score (0=confident, 1=likely hallucinating), metrics, contributions, confidence.
    """
    default_weights = {
        "entropy": 0.25,
        "token_prob": 0.25,
        "consistency": 0.20,
        "sparsity": 0.15,
        "lookback": 0.15,
    }
    if weights is not None:
        default_weights.update(weights)
    w = default_weights

    metrics = {}
    contributions = {}
    total_weight = 0.0
    weighted_sum = 0.0

    if attn is not None:
        attn_np = np.asarray(attn, dtype=np.float64)
        ent = attention_entropy(attn_np)
        seq_len = attn_np.shape[-1]
        max_entropy = np.log(max(seq_len, 2))
        norm_ent = float(np.mean(ent) / max(max_entropy, 1e-12))
        norm_ent = np.clip(norm_ent, 0.0, 1.0)
        metrics["entropy"] = norm_ent
        contributions["entropy"] = norm_ent * w["entropy"]
        weighted_sum += contributions["entropy"]
        total_weight += w["entropy"]

    if token_probs is not None:
        tp_stats = token_probability_stats(token_probs)
        tp_signal = 1.0 - tp_stats["mean"]
        tp_signal = np.clip(tp_signal, 0.0, 1.0)
        metrics["token_prob"] = tp_stats
        contributions["token_prob"] = float(tp_signal) * w["token_prob"]
        weighted_sum += contributions["token_prob"]
        total_weight += w["token_prob"]

    if layers is not None and len(layers) >= 2:
        corrs = cross_layer_consistency(layers)
        mean_corr = float(np.mean(corrs))
        consist_signal = 1.0 - max(mean_corr, 0.0)
        consist_signal = np.clip(consist_signal, 0.0, 1.0)
        metrics["consistency"] = float(mean_corr)
        contributions["consistency"] = consist_signal * w["consistency"]
        weighted_sum += contributions["consistency"]
        total_weight += w["consistency"]

    if attn is not None:
        attn_np = np.asarray(attn, dtype=np.float64)
        sp = attention_sparsity(attn_np)
        mean_sp = float(np.mean(sp))
        mean_sp = np.clip(mean_sp, 0.0, 1.0)
        metrics["sparsity"] = mean_sp
        contributions["sparsity"] = mean_sp * w["sparsity"]
        weighted_sum += contributions["sparsity"]
        total_weight += w["sparsity"]

    if attn is not None and context_len is not None and context_len > 0:
        attn_np = np.asarray(attn, dtype=np.float64)
        lb = lookback_ratio(attn_np, context_len)
        lb_signal = 1.0 - lb
        lb_signal = np.clip(lb_signal, 0.0, 1.0)
        metrics["lookback"] = lb
        contributions["lookback"] = float(lb_signal) * w["lookback"]
        weighted_sum += contributions["lookback"]
        total_weight += w["lookback"]

    if total_weight > 0:
        score = weighted_sum / total_weight
    else:
        score = 0.5

    confidence = total_weight / sum(w.values())

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "metrics": metrics,
        "contributions": contributions,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }
