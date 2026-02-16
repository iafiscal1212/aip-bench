"""
Bench Evaluator: Benchmark metrics (numpy-pure, no sklearn).

AUROC, F1, ECE, Brier, Perplexity, BLEU, ROUGE-L, Accuracy,
Exact Match, Token Efficiency, and QA metrics for evaluating
hallucination detection and inference optimization.

Author: Carmen Esteban
"""

import re
import string

import numpy as np


# ============================================================
# Internal helpers
# ============================================================

def _normalize_answer(text):
    """Normalize answer for SQuAD-style evaluation.

    Lowercase, strip, remove articles (a/an/the), remove punctuation,
    collapse whitespace.
    """
    text = str(text).lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


# ============================================================
# Classification metrics
# ============================================================

def auroc_score(y_true, y_scores):
    """Area Under the ROC Curve via Wilcoxon-Mann-Whitney statistic.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0 or 1).
    y_scores : array-like
        Predicted scores (higher = more likely positive).

    Returns
    -------
    float
        AUROC in [0, 1]. Returns 0.5 if only one class present.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Wilcoxon-Mann-Whitney via ranks
    n = len(y_scores)
    # Rank scores (average rank for ties)
    order = np.argsort(y_scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Handle ties: assign average rank to tied values
    sorted_scores = y_scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    # AUROC = (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    rank_sum = np.sum(ranks[pos_mask])
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def precision_recall(y_true, y_pred):
    """Precision, recall, and confusion matrix counts.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_pred : array-like
        Binary predicted labels.

    Returns
    -------
    dict
        {precision, recall, tp, fp, fn, tn}
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def expected_calibration_error(y_true, y_scores, n_bins=10):
    """Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    Low ECE means the model "knows when it doesn't know."

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_scores : array-like
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of bins for calibration. Default 10.

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better calibrated.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(y_true) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_scores >= lo) & (y_scores <= hi)
        else:
            mask = (y_scores >= lo) & (y_scores < hi)
        n_bin = np.sum(mask)
        if n_bin == 0:
            continue
        avg_confidence = float(np.mean(y_scores[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def abstention_rate(y_scores, low_threshold=0.3, high_threshold=0.7):
    """Abstention rate: fraction of predictions in the uncertain zone.

    When the model's score falls between low_threshold and high_threshold,
    it is uncertain and should abstain rather than hallucinate.

    Parameters
    ----------
    y_scores : array-like
        Predicted scores in [0, 1].
    low_threshold : float
        Below this, confident negative. Default 0.3.
    high_threshold : float
        Above this, confident positive. Default 0.7.

    Returns
    -------
    dict
        {abstention_rate, n_abstained, n_confident, n_total}
    """
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
    n_total = len(y_scores)
    if n_total == 0:
        return {"abstention_rate": 0.0, "n_abstained": 0,
                "n_confident": 0, "n_total": 0}

    uncertain = (y_scores >= low_threshold) & (y_scores <= high_threshold)
    n_abstained = int(np.sum(uncertain))
    n_confident = n_total - n_abstained

    return {
        "abstention_rate": float(n_abstained / n_total),
        "n_abstained": n_abstained,
        "n_confident": n_confident,
        "n_total": n_total,
    }


def brier_score(y_true, y_scores):
    """Brier score: mean squared error of predicted probabilities.

    Lower is better. Decomposable into calibration + refinement.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_scores : array-like
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Brier score in [0, 1]. 0 = perfect, 1 = worst.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_scores - y_true) ** 2))


def perplexity(log_probs):
    """Perplexity from per-token log probabilities.

    Parameters
    ----------
    log_probs : array-like
        Log probabilities of each token (negative values).

    Returns
    -------
    float
        Perplexity. Lower = model is more confident/accurate.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64).ravel()
    if len(log_probs) == 0:
        return float("inf")
    avg_neg_log = -float(np.mean(log_probs))
    return float(np.exp(min(avg_neg_log, 700)))  # clip to avoid overflow


def accuracy_score(y_true, y_pred):
    """Classification accuracy (works for multiclass).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def bleu_score(prediction, reference, max_n=4):
    """Corpus-level BLEU score (single pair, simplified).

    Parameters
    ----------
    prediction : str
        Generated text.
    reference : str
        Reference text.
    max_n : int
        Maximum n-gram order. Default 4.

    Returns
    -------
    float
        BLEU score in [0, 1].
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1)))

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        if len(pred_ngrams) == 0:
            precisions.append(0.0)
            continue
        matches = sum(min(pred_ngrams[ng], ref_ngrams.get(ng, 0))
                      for ng in pred_ngrams)
        precisions.append(matches / sum(pred_ngrams.values()))

    if any(p == 0.0 for p in precisions):
        return 0.0

    log_avg = sum(np.log(max(p, 1e-12)) for p in precisions) / max_n
    return float(bp * np.exp(log_avg))


def rouge_l_score(prediction, reference):
    """ROUGE-L score (longest common subsequence F1).

    Parameters
    ----------
    prediction : str
        Generated text.
    reference : str
        Reference text.

    Returns
    -------
    float
        ROUGE-L F1 in [0, 1].
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    prec = lcs_len / len(pred_tokens)
    rec = lcs_len / len(ref_tokens)
    if prec + rec < 1e-12:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _get_ngrams(tokens, n):
    """Count n-grams in a token list."""
    counts = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _lcs_length(a, b):
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimized: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def f1_score(y_true, y_pred):
    """F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_pred : array-like
        Binary predicted labels.

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    pr = precision_recall(y_true, y_pred)
    p, r = pr["precision"], pr["recall"]
    if p + r < 1e-12:
        return 0.0
    return float(2 * p * r / (p + r))


# ============================================================
# QA metrics
# ============================================================

def exact_match(predictions, references):
    """Exact match score after SQuAD normalization.

    Parameters
    ----------
    predictions : list of str
        Predicted answers.
    references : list of str
        Ground truth answers.

    Returns
    -------
    float
        Fraction of exact matches in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references)
        if _normalize_answer(p) == _normalize_answer(r)
    )
    return float(matches / len(predictions))


def f1_score_qa(prediction, reference):
    """Token-level F1 score for a single QA pair.

    Parameters
    ----------
    prediction : str
        Predicted answer.
    reference : str
        Ground truth answer.

    Returns
    -------
    float
        Token-level F1 in [0, 1].
    """
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    prec = num_common / len(pred_tokens)
    rec = num_common / len(ref_tokens)
    return float(2 * prec * rec / (prec + rec))


def qa_metrics(predictions, references):
    """Compute QA metrics (EM and F1) over a dataset.

    Parameters
    ----------
    predictions : list of str
        Predicted answers.
    references : list of str
        Ground truth answers.

    Returns
    -------
    dict
        {exact_match, f1, n}
    """
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "f1": 0.0, "n": 0}

    em = exact_match(predictions, references)
    f1s = [f1_score_qa(p, r) for p, r in zip(predictions, references)]
    return {
        "exact_match": em,
        "f1": float(np.mean(f1s)),
        "n": n,
    }


# ============================================================
# Efficiency metrics
# ============================================================

def token_efficiency(accuracy, tokens_used, token_budget):
    """Token efficiency metric for reasoning benchmarks.

    efficiency = accuracy * (1 - tokens_used / budget)

    Parameters
    ----------
    accuracy : float
        Task accuracy in [0, 1].
    tokens_used : float
        Number of tokens used.
    token_budget : float
        Maximum token budget.

    Returns
    -------
    dict
        {efficiency, accuracy, tokens_used, budget_utilization,
         tokens_saved, savings_ratio}
    """
    budget_util = tokens_used / max(token_budget, 1)
    tokens_saved = max(token_budget - tokens_used, 0)
    savings_ratio = tokens_saved / max(token_budget, 1)
    eff = accuracy * max(1.0 - budget_util, 0.0)

    return {
        "efficiency": float(eff),
        "accuracy": float(accuracy),
        "tokens_used": float(tokens_used),
        "budget_utilization": float(budget_util),
        "tokens_saved": float(tokens_saved),
        "savings_ratio": float(savings_ratio),
    }


def input_compression_efficiency(accuracy_full, accuracy_compressed,
                                 tokens_full, tokens_compressed):
    """Input compression efficiency: same quality with fewer input tokens.

    Measures whether compressing the input prompt preserves answer quality.

    Parameters
    ----------
    accuracy_full : float
        Accuracy with full (uncompressed) input.
    accuracy_compressed : float
        Accuracy with compressed input.
    tokens_full : float
        Input tokens before compression.
    tokens_compressed : float
        Input tokens after compression.

    Returns
    -------
    dict
        {quality_ratio, compression_ratio, input_efficiency,
         tokens_full, tokens_compressed}
    """
    quality_ratio = accuracy_compressed / max(accuracy_full, 1e-12)
    compression_ratio = 1.0 - tokens_compressed / max(tokens_full, 1)
    # efficiency = quality_preserved * tokens_saved
    input_eff = quality_ratio * max(compression_ratio, 0.0)

    return {
        "quality_ratio": float(quality_ratio),
        "compression_ratio": float(compression_ratio),
        "input_efficiency": float(input_eff),
        "tokens_full": float(tokens_full),
        "tokens_compressed": float(tokens_compressed),
    }


def output_quality_per_token(accuracy, tokens_generated):
    """Output quality per token: accuracy normalized by generation length.

    Measures how much useful information each generated token carries.

    Parameters
    ----------
    accuracy : float
        Task accuracy or quality score in [0, 1].
    tokens_generated : float
        Number of tokens generated in the output.

    Returns
    -------
    dict
        {quality_per_token, accuracy, tokens_generated}
    """
    qpt = accuracy / max(tokens_generated, 1)

    return {
        "quality_per_token": float(qpt),
        "accuracy": float(accuracy),
        "tokens_generated": float(tokens_generated),
    }


# ============================================================
# Threshold optimization
# ============================================================

def optimal_threshold(y_true, y_scores, metric="f1"):
    """Find threshold that maximizes a metric.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_scores : array-like
        Predicted scores.
    metric : str
        Metric to optimize. Currently only 'f1' supported.

    Returns
    -------
    dict
        {threshold, score, metric}
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    thresholds = np.unique(y_scores)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_scores >= t).astype(np.int64)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return {
        "threshold": best_threshold,
        "score": float(best_score),
        "metric": metric,
    }


# ============================================================
# BenchmarkResult container
# ============================================================

def bootstrap_ci(metric_fn, y_true, y_scores, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for any metric function.

    Parameters
    ----------
    metric_fn : callable
        Function(y_true, y_scores) -> float.
    y_true : array-like
        Ground truth.
    y_scores : array-like
        Predictions or scores.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95%).
    seed : int
        Random seed.

    Returns
    -------
    dict
        {mean, ci_low, ci_high, ci, n_bootstrap}
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores.append(float(metric_fn(y_true[idx], y_scores[idx])))
    scores.sort()
    alpha = (1 - ci) / 2
    lo = scores[max(0, int(alpha * n_bootstrap))]
    hi = scores[min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap))]
    return {
        "mean": float(np.mean(scores)),
        "ci_low": lo,
        "ci_high": hi,
        "ci": ci,
        "n_bootstrap": n_bootstrap,
    }


def macro_f1(y_true, y_pred):
    """Macro-averaged F1 score for multiclass classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (any type).
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Macro F1 in [0, 1].
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        binary_true = (y_true == c).astype(np.int64)
        binary_pred = (y_pred == c).astype(np.int64)
        f1s.append(f1_score(binary_true, binary_pred))
    if len(f1s) == 0:
        return 0.0
    return float(np.mean(f1s))


def pass_at_k(n, c, k):
    """pass@k metric for code generation evaluation.

    Parameters
    ----------
    n : int
        Total number of generated samples.
    c : int
        Number of correct samples.
    k : int
        k in pass@k.

    Returns
    -------
    float
        pass@k probability in [0, 1].
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def meteor_score(prediction, reference):
    """Simplified METEOR score (unigram F-mean with fragmentation penalty).

    Parameters
    ----------
    prediction : str
        Generated text.
    reference : str
        Reference text.

    Returns
    -------
    float
        METEOR score in [0, 1].
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Unigram matches
    ref_set = set(ref_tokens)
    matches = sum(1 for t in pred_tokens if t in ref_set)
    if matches == 0:
        return 0.0

    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)

    # Weighted F-mean (alpha=0.9 as in original METEOR)
    alpha = 0.9
    denom = alpha * precision + (1 - alpha) * recall
    if denom < 1e-12:
        return 0.0
    f_mean = (precision * recall) / denom

    # Fragmentation penalty: count contiguous matched chunks
    chunks = 0
    in_match = False
    for t in pred_tokens:
        if t in ref_set:
            if not in_match:
                chunks += 1
                in_match = True
        else:
            in_match = False

    frag = chunks / matches
    penalty = 0.5 * (frag ** 3)
    return float(f_mean * (1 - penalty))


def paired_permutation_test(scores_a, scores_b, n_permutations=10000, seed=42):
    """Paired permutation test for comparing two systems.

    Tests whether system A is significantly different from system B.

    Parameters
    ----------
    scores_a : array-like
        Per-sample scores from system A.
    scores_b : array-like
        Per-sample scores from system B.
    n_permutations : int
        Number of random permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {observed_diff, p_value, n_permutations, significant_005, significant_001}
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    assert len(scores_a) == len(scores_b), "Arrays must have same length"

    diffs = scores_a - scores_b
    observed = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    n = len(diffs)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = float(np.mean(diffs * signs))
        if abs(perm_diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_diff": observed,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def effect_size_cohens_d(scores_a, scores_b):
    """Cohen's d effect size between two score distributions.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Score arrays.

    Returns
    -------
    dict
        {d, magnitude} where magnitude is 'negligible'/'small'/'medium'/'large'.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    n_a, n_b = len(scores_a), len(scores_b)
    var_a = np.var(scores_a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(scores_b, ddof=1) if n_b > 1 else 0.0

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))

    if pooled_std < 1e-12:
        d = 0.0
    else:
        d = float((np.mean(scores_a) - np.mean(scores_b)) / pooled_std)

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {"d": d, "magnitude": magnitude}


class BenchmarkResult:
    """Container for benchmark results.

    Parameters
    ----------
    name : str
        Benchmark name (e.g. 'halueval', 'ockbench', 'qa_compression').
    metrics : dict
        Dictionary of metric name -> value.
    metadata : dict, optional
        Additional metadata (dataset size, parameters, etc.).
    """

    def __init__(self, name, metrics, metadata=None):
        self.name = name
        self.metrics = dict(metrics)
        self.metadata = dict(metadata) if metadata else {}

    def to_dict(self):
        """Convert to plain dictionary."""
        return {
            "name": self.name,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def summary(self):
        """Human-readable summary string."""
        lines = [f"Benchmark: {self.name}"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        if self.metadata:
            lines.append("  ---")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def __repr__(self):
        metric_strs = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.metrics.items()
        )
        return f"BenchmarkResult({self.name!r}, {{{metric_strs}}})"
