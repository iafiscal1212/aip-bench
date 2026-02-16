"""
Bench Pipelines: Benchmark runners connecting guard/inference with evaluator.

Seven pipeline types:
1. HallucinationBenchmark — AUROC, F1 for hallucination detection
2. InferenceEfficiencyBenchmark — Token efficiency for reasoning
3. QACompressionBenchmark — Quality retention under KV compression
4. MultipleChoiceBenchmark — MMLU, HellaSwag, ARC, WinoGrande, BoolQ
5. MathBenchmark — GSM8K math reasoning
6. FactVerificationBenchmark — FEVER fact verification
7. OpenDomainQABenchmark — NQ, SQuAD, HotPotQA

Author: Carmen Esteban
"""

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _progress(iterable, desc=None, total=None):
    """Wrap iterable with tqdm if available, otherwise passthrough."""
    if HAS_TQDM:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable

from aip_bench.guard import hallucination_score
from aip_bench.inference import estimate_savings, compress_kv_cache
from aip_bench.evaluator import (
    auroc_score,
    f1_score,
    precision_recall,
    optimal_threshold,
    token_efficiency,
    qa_metrics,
    accuracy_score,
    macro_f1,
    exact_match,
    BenchmarkResult,
)
from aip_bench.datasets import (
    SyntheticHaluEval,
    SyntheticOckBench,
    SyntheticQA,
)
from aip_bench.logging_utils import get_logger

_log = get_logger(__name__)


class HallucinationBenchmark:
    """Benchmark hallucination detection against labeled data.

    Runs hallucination_score() on each sample and evaluates
    AUROC, optimal threshold, F1, precision, and recall.

    Parameters
    ----------
    data : iterable
        Samples with {attn, token_probs, layers, context_len, label}.
    weights : dict, optional
        Override default hallucination_score weights.
    """

    def __init__(self, data=None, weights=None):
        if data is None:
            data = SyntheticHaluEval()
        self.data = data
        self.weights = weights

    def run(self):
        """Run the benchmark.

        Returns
        -------
        BenchmarkResult
            With metrics: auroc, f1, precision, recall, optimal_threshold.
        """
        labels = []
        scores = []

        for sample in _progress(self.data, desc="HaluEval"):
            label = sample["label"]
            result = hallucination_score(
                attn=sample.get("attn"),
                token_probs=sample.get("token_probs"),
                layers=sample.get("layers"),
                context_len=sample.get("context_len"),
                weights=self.weights,
            )
            labels.append(label)
            scores.append(result["score"])

        labels = np.array(labels, dtype=np.int64)
        scores = np.array(scores, dtype=np.float64)

        auroc = auroc_score(labels, scores)
        opt = optimal_threshold(labels, scores, metric="f1")
        preds = (scores >= opt["threshold"]).astype(np.int64)
        f1 = f1_score(labels, preds)
        pr = precision_recall(labels, preds)

        metrics = {
            "auroc": auroc,
            "f1": f1,
            "precision": pr["precision"],
            "recall": pr["recall"],
            "optimal_threshold": opt["threshold"],
        }
        metadata = {
            "n_samples": len(labels),
            "n_positive": int(np.sum(labels)),
            "n_negative": int(np.sum(labels == 0)),
        }

        return BenchmarkResult("halueval", metrics, metadata)


class InferenceEfficiencyBenchmark:
    """Benchmark inference efficiency via head pruning + KV compression.

    Runs estimate_savings() on each problem and computes
    token efficiency metrics.

    Parameters
    ----------
    data : iterable
        Problems with {attn, keys, values, correct, tokens_used}.
    prune_ratio : float
        Fraction of heads to prune.
    compress_method : str
        KV compression method.
    token_budget : int
        Token budget for efficiency calculation.
    """

    def __init__(self, data=None, prune_ratio=0.25,
                 compress_method="evict", token_budget=1024):
        if data is None:
            data = SyntheticOckBench()
        self.data = data
        self.prune_ratio = prune_ratio
        self.compress_method = compress_method
        self.token_budget = token_budget

    def run(self):
        """Run the benchmark.

        Returns
        -------
        BenchmarkResult
            With metrics: accuracy, efficiency, tokens_saved, savings_ratio,
            mean_effective_tokens, mean_combined_savings.
        """
        effective_tokens_list = []
        correct_list = []

        for problem in _progress(self.data, desc="OckBench"):
            savings = estimate_savings(
                attn=problem["attn"],
                keys=problem["keys"],
                values=problem["values"],
                prune_ratio=self.prune_ratio,
                compress_method=self.compress_method,
            )

            # Combined savings from both techniques
            param_savings = savings["total_param_reduction_ratio"]
            mem_savings = savings["total_memory_reduction_ratio"]
            combined = 1.0 - (1.0 - param_savings) * (1.0 - mem_savings)

            effective = problem["tokens_used"] * (1.0 - combined)
            effective_tokens_list.append(effective)
            correct_list.append(problem["correct"])

        effective_tokens = np.array(effective_tokens_list)
        correct = np.array(correct_list)

        accuracy = float(np.mean(correct))
        mean_effective = float(np.mean(effective_tokens))
        mean_combined = float(1.0 - np.mean(effective_tokens) /
                              max(np.mean([p["tokens_used"] for p in self.data]), 1))

        eff = token_efficiency(accuracy, mean_effective, self.token_budget)

        metrics = {
            "accuracy": accuracy,
            "efficiency": eff["efficiency"],
            "tokens_saved": eff["tokens_saved"],
            "savings_ratio": eff["savings_ratio"],
            "mean_effective_tokens": mean_effective,
            "mean_combined_savings": mean_combined,
        }
        metadata = {
            "n_problems": len(correct_list),
            "prune_ratio": self.prune_ratio,
            "compress_method": self.compress_method,
            "token_budget": self.token_budget,
        }

        return BenchmarkResult("ockbench", metrics, metadata)


class QACompressionBenchmark:
    """Benchmark QA quality retention under KV cache compression.

    Compresses key-value caches and measures quality degradation
    via cosine similarity between original and compressed caches.

    Parameters
    ----------
    data : iterable
        Samples with {question, context, answer, keys, values}.
    compress_method : str
        KV compression method.
    recent_window : int
        Recent window for compression.
    """

    def __init__(self, data=None, compress_method="evict", recent_window=64):
        if data is None:
            data = SyntheticQA()
        self.data = data
        self.compress_method = compress_method
        self.recent_window = recent_window

    def run(self):
        """Run the benchmark.

        Returns
        -------
        BenchmarkResult
            With metrics: em_full, em_compressed, f1_full, f1_compressed,
            quality_retention, mean_cosine_similarity.
        """
        answers_full = []
        answers_compressed = []
        references = []
        cosine_sims = []

        for sample in _progress(self.data, desc="HaluEval"):
            keys = np.asarray(sample["keys"], dtype=np.float64)
            values = np.asarray(sample["values"], dtype=np.float64)
            answer = sample["answer"]

            # Full (uncompressed) "answer" — use the original answer
            answers_full.append(answer)
            references.append(answer)

            # Compress KV cache
            result = compress_kv_cache(
                keys, values,
                method=self.compress_method,
                recent_window=self.recent_window,
            )

            # Measure cosine similarity between original and compressed
            comp_keys = np.asarray(result["keys"], dtype=np.float64)
            if comp_keys.ndim == 2:
                comp_len = comp_keys.shape[0]
                orig_len = keys.shape[0]

                # For evict: compressed = last N tokens of original
                # Compare the recent window of original with compressed
                if comp_len < orig_len:
                    orig_recent = keys[orig_len - comp_len:]
                    orig_flat = orig_recent.ravel()
                else:
                    orig_flat = keys[:comp_len].ravel()
                comp_flat = comp_keys.ravel()

                dot = np.dot(orig_flat, comp_flat)
                norm_o = np.linalg.norm(orig_flat)
                norm_c = np.linalg.norm(comp_flat)
                if norm_o > 1e-12 and norm_c > 1e-12:
                    cos_sim = dot / (norm_o * norm_c)
                else:
                    cos_sim = 0.0
                # Clamp to [0, 1] for quality estimation
                cos_sim = max(0.0, float(cos_sim))
                cosine_sims.append(cos_sim)

                # Simulate compressed answer degradation based on similarity
                # High similarity -> answer preserved; low -> degraded
                if cos_sim > 0.8:
                    answers_compressed.append(answer)
                elif cos_sim > 0.4:
                    # Partial degradation
                    words = answer.split()
                    keep = max(1, int(len(words) * (0.5 + cos_sim * 0.5)))
                    answers_compressed.append(" ".join(words[:keep]))
                else:
                    words = answer.split()
                    keep = max(1, int(len(words) * 0.3))
                    answers_compressed.append(" ".join(words[:keep]))
            else:
                # Sparse format — use original answer
                answers_compressed.append(answer)
                cosine_sims.append(1.0)

        # Compute QA metrics
        metrics_full = qa_metrics(answers_full, references)
        metrics_compressed = qa_metrics(answers_compressed, references)

        f1_full = metrics_full["f1"]
        f1_comp = metrics_compressed["f1"]
        quality_retention = f1_comp / max(f1_full, 1e-12)

        metrics = {
            "em_full": metrics_full["exact_match"],
            "em_compressed": metrics_compressed["exact_match"],
            "f1_full": f1_full,
            "f1_compressed": f1_comp,
            "quality_retention": float(quality_retention),
            "mean_cosine_similarity": float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        }
        metadata = {
            "n_samples": len(references),
            "compress_method": self.compress_method,
            "recent_window": self.recent_window,
        }

        return BenchmarkResult("qa_compression", metrics, metadata)


# ============================================================
# New pipelines: MC, Math, FactVerification, OpenDomainQA
# ============================================================

class MultipleChoiceBenchmark:
    """Benchmark for multiple-choice tasks (MMLU, HellaSwag, ARC, WinoGrande, BoolQ).

    Parameters
    ----------
    task : str
        Task name for prompt template.
    data : list of dict, optional
        Items with {question, choices, answer_idx, ...}.
        If None, generates synthetic data.
    model : BaseModel, optional
        Model backend. Defaults to DummyModel.
    n_shots : int
        Number of few-shot examples.
    examples : list of dict, optional
        Few-shot example pool.
    """

    def __init__(self, task="mmlu", data=None, model=None,
                 n_shots=0, examples=None):
        self.task = task
        self.data = data
        self.model = model
        self.n_shots = n_shots
        self.examples = examples

    def run(self):
        from aip_bench.models import DummyModel
        from aip_bench.prompts import format_prompt

        if self.model is None:
            self.model = DummyModel()
        if self.data is None:
            self.data = _synthetic_mc(50)

        predictions = []
        references = []

        for item in _progress(self.data, desc=self.task):
            prompt = format_prompt(
                self.task, item,
                n_shots=self.n_shots, examples=self.examples,
            )
            choices = item.get("choices", ["A", "B", "C", "D"])
            pred_idx = self.model.classify(prompt, choices)
            predictions.append(pred_idx)
            references.append(item.get("answer_idx", 0))

        predictions = np.array(predictions)
        references = np.array(references)
        acc = float(accuracy_score(references, predictions))
        mf1 = float(macro_f1(references, predictions))

        return BenchmarkResult(self.task, {
            "accuracy": acc,
            "macro_f1": mf1,
            "n_correct": int(np.sum(predictions == references)),
            "n_total": len(predictions),
        }, {
            "n_shots": self.n_shots,
            "model": self.model.name,
        })


class MathBenchmark:
    """Benchmark for math tasks (GSM8K).

    Parameters
    ----------
    task : str
        Task name.
    data : list of dict, optional
        Items with {question, answer}.
    model : BaseModel, optional
        Model backend.
    n_shots : int
        Number of few-shot examples.
    examples : list of dict, optional
        Few-shot examples.
    """

    def __init__(self, task="gsm8k", data=None, model=None,
                 n_shots=0, examples=None):
        self.task = task
        self.data = data
        self.model = model
        self.n_shots = n_shots
        self.examples = examples

    def run(self):
        from aip_bench.models import DummyModel
        from aip_bench.prompts import format_prompt

        if self.model is None:
            self.model = DummyModel(default_response="42")
        if self.data is None:
            self.data = _synthetic_math(50)

        predictions = []
        references = []

        for item in _progress(self.data, desc=self.task):
            prompt = format_prompt(
                self.task, item,
                n_shots=self.n_shots, examples=self.examples,
            )
            response = self.model.generate(prompt)
            pred_num = _extract_number(response)
            ref_num = _extract_number(str(item.get("answer", "")))
            predictions.append(pred_num)
            references.append(ref_num)

        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        total = len(predictions)
        acc = correct / max(total, 1)

        return BenchmarkResult(self.task, {
            "accuracy": float(acc),
            "n_correct": correct,
            "n_total": total,
        }, {
            "n_shots": self.n_shots,
            "model": self.model.name,
        })


class FactVerificationBenchmark:
    """Benchmark for fact verification tasks (FEVER).

    Parameters
    ----------
    task : str
        Task name.
    data : list of dict, optional
        Items with {claim, evidence, label}.
    model : BaseModel, optional
        Model backend.
    n_shots : int
        Number of few-shot examples.
    examples : list of dict, optional
        Few-shot examples.
    """

    def __init__(self, task="fever", data=None, model=None,
                 n_shots=0, examples=None):
        self.task = task
        self.data = data
        self.model = model
        self.n_shots = n_shots
        self.examples = examples

    def run(self):
        from aip_bench.models import DummyModel
        from aip_bench.prompts import format_prompt

        if self.model is None:
            self.model = DummyModel(default_response="SUPPORTS")
        if self.data is None:
            self.data = _synthetic_fever(50)

        labels_map = {"supports": 0, "refutes": 1, "not enough info": 2}
        predictions = []
        references = []

        for item in _progress(self.data, desc=self.task):
            prompt = format_prompt(
                self.task, item,
                n_shots=self.n_shots, examples=self.examples,
            )
            response = self.model.generate(prompt).strip().lower()
            pred = _classify_verdict(response)
            ref = labels_map.get(str(item.get("label", "")).lower(), 2)
            predictions.append(pred)
            references.append(ref)

        predictions = np.array(predictions)
        references = np.array(references)
        acc = float(accuracy_score(references, predictions))
        mf1 = float(macro_f1(references, predictions))

        return BenchmarkResult(self.task, {
            "accuracy": acc,
            "macro_f1": mf1,
            "n_correct": int(np.sum(predictions == references)),
            "n_total": len(predictions),
        }, {
            "n_shots": self.n_shots,
            "model": self.model.name,
        })


class OpenDomainQABenchmark:
    """Benchmark for open-domain QA tasks (NQ, SQuAD, HotPotQA).

    Parameters
    ----------
    task : str
        Task name.
    data : list of dict, optional
        Items with {question, context, answer}.
    model : BaseModel, optional
        Model backend.
    n_shots : int
        Number of few-shot examples.
    examples : list of dict, optional
        Few-shot examples.
    """

    def __init__(self, task="natural_questions", data=None, model=None,
                 n_shots=0, examples=None):
        self.task = task
        self.data = data
        self.model = model
        self.n_shots = n_shots
        self.examples = examples

    def run(self):
        from aip_bench.models import DummyModel
        from aip_bench.prompts import format_prompt

        if self.model is None:
            self.model = DummyModel()
        if self.data is None:
            self.data = _synthetic_open_qa(50)

        predictions = []
        references = []

        for item in _progress(self.data, desc=self.task):
            prompt = format_prompt(
                self.task, item,
                n_shots=self.n_shots, examples=self.examples,
            )
            response = self.model.generate(prompt).strip()
            predictions.append(response)
            references.append(str(item.get("answer", "")))

        em = exact_match(predictions, references)
        qam = qa_metrics(predictions, references)

        return BenchmarkResult(self.task, {
            "exact_match": em,
            "f1": qam["f1"],
            "n_total": qam["n"],
        }, {
            "n_shots": self.n_shots,
            "model": self.model.name,
        })


# ============================================================
# Synthetic data generators for new pipelines
# ============================================================

def _synthetic_mc(n):
    """Generate synthetic multiple-choice items."""
    rng = np.random.RandomState(42)
    items = []
    subjects = ["math", "science", "history", "literature"]
    for i in range(n):
        items.append({
            "question": f"Synthetic question {i}?",
            "choices": [f"Option {j}" for j in range(4)],
            "answer_idx": int(rng.randint(0, 4)),
            "subject": subjects[i % len(subjects)],
            "A": "Option 0", "B": "Option 1",
            "C": "Option 2", "D": "Option 3",
        })
    return items


def _synthetic_math(n):
    """Generate synthetic math problems."""
    rng = np.random.RandomState(42)
    items = []
    for i in range(n):
        a = int(rng.randint(1, 100))
        b = int(rng.randint(1, 100))
        items.append({
            "question": f"What is {a} + {b}?",
            "answer": str(a + b),
        })
    return items


def _synthetic_fever(n):
    """Generate synthetic fact verification items."""
    rng = np.random.RandomState(42)
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    items = []
    for i in range(n):
        items.append({
            "claim": f"Synthetic claim {i}.",
            "evidence": f"Synthetic evidence for claim {i}.",
            "label": labels[int(rng.randint(0, 3))],
            "verdict": labels[int(rng.randint(0, 3))],
        })
    return items


def _synthetic_open_qa(n):
    """Generate synthetic open-domain QA items."""
    items = []
    for i in range(n):
        items.append({
            "question": f"What is synthetic item {i}?",
            "context": f"Synthetic item {i} is the answer.",
            "answer": "the answer",
        })
    return items


# ============================================================
# Helpers
# ============================================================

def _extract_number(text):
    """Extract the last number from a string (for math answers)."""
    import re
    # Remove commas from numbers
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def _classify_verdict(text):
    """Map free-text verdict to label index."""
    text = text.lower()
    if "support" in text:
        return 0
    elif "refute" in text:
        return 1
    else:
        return 2


# ============================================================
# Dispatcher
# ============================================================

_BENCHMARKS = {
    # Original pipelines
    "halueval": HallucinationBenchmark,
    "ockbench": InferenceEfficiencyBenchmark,
    "qa_compression": QACompressionBenchmark,
    # New pipelines
    "mmlu": MultipleChoiceBenchmark,
    "hellaswag": MultipleChoiceBenchmark,
    "arc_challenge": MultipleChoiceBenchmark,
    "winogrande": MultipleChoiceBenchmark,
    "boolq": MultipleChoiceBenchmark,
    "gsm8k": MathBenchmark,
    "fever": FactVerificationBenchmark,
    "natural_questions": OpenDomainQABenchmark,
    "squad_v2": OpenDomainQABenchmark,
    "hotpotqa": OpenDomainQABenchmark,
    "truthfulqa": MultipleChoiceBenchmark,
}

# Map task name to its default task parameter
_TASK_DEFAULTS = {
    "mmlu": {"task": "mmlu"},
    "hellaswag": {"task": "hellaswag"},
    "arc_challenge": {"task": "arc_challenge"},
    "winogrande": {"task": "winogrande"},
    "boolq": {"task": "boolq"},
    "gsm8k": {"task": "gsm8k"},
    "fever": {"task": "fever"},
    "natural_questions": {"task": "natural_questions"},
    "squad_v2": {"task": "squad_v2"},
    "hotpotqa": {"task": "hotpotqa"},
    "truthfulqa": {"task": "truthfulqa"},
}


def run_benchmark(name, use_synthetic=True, **kwargs):
    """Run a named benchmark.

    Parameters
    ----------
    name : str
        Benchmark name (e.g. 'halueval', 'mmlu', 'gsm8k').
    use_synthetic : bool
        If True, use synthetic data (default).
    **kwargs
        Additional arguments passed to the benchmark constructor.

    Returns
    -------
    BenchmarkResult
        Benchmark results with metrics and metadata.

    Raises
    ------
    ValueError
        If the benchmark name is not recognized.
    """
    if name not in _BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {name!r}. "
            f"Available: {list(_BENCHMARKS.keys())}"
        )

    cls = _BENCHMARKS[name]
    # Inject task defaults for new-style pipelines
    if name in _TASK_DEFAULTS:
        for k, v in _TASK_DEFAULTS[name].items():
            kwargs.setdefault(k, v)
    _log.info("Running benchmark: %s", name)
    benchmark = cls(**kwargs)
    result = benchmark.run()
    _log.debug("Benchmark %s complete", name)
    return result
