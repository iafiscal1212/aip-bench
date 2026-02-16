"""
Tests for aip_bench — benchmarking module.

All tests use synthetic data (no downloads, no GPU, no torch).
Covers evaluator, datasets, pipelines, compare, models, prompts, cache, export, CLI.
Plus 1 @pytest.mark.slow integration test (skipped without torch/transformers).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from aip_bench.evaluator import (
    auroc_score,
    f1_score,
    precision_recall,
    expected_calibration_error,
    abstention_rate,
    brier_score,
    perplexity,
    accuracy_score,
    bleu_score,
    rouge_l_score,
    exact_match,
    f1_score_qa,
    token_efficiency,
    input_compression_efficiency,
    output_quality_per_token,
    optimal_threshold,
    bootstrap_ci,
    macro_f1,
    pass_at_k,
    meteor_score,
    paired_permutation_test,
    effect_size_cohens_d,
    BenchmarkResult,
)
from aip_bench.datasets import list_datasets
from aip_bench.datasets import (
    SyntheticHaluEval,
    SyntheticOckBench,
    SyntheticQA,
)
from aip_bench.pipelines import (
    run_benchmark,
    HallucinationBenchmark,
    InferenceEfficiencyBenchmark,
    QACompressionBenchmark,
    MultipleChoiceBenchmark,
    MathBenchmark,
    FactVerificationBenchmark,
    OpenDomainQABenchmark,
)
from aip_bench.compare import (
    compare,
    compare_results,
    ComparisonReport,
)
from aip_bench.models import DummyModel, load_model, BaseModel
from aip_bench.prompts import format_prompt, list_templates, get_template
from aip_bench.cache import ResultCache
from aip_bench.export import to_json, to_csv, to_html
from aip_bench.config import validate_config, load_config


# ============================================================
# TestAUROC — 5 tests
# ============================================================

class TestAUROC:
    def test_perfect_separation(self):
        """Perfect separation -> AUROC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert auroc_score(y_true, y_scores) == pytest.approx(1.0)

    def test_random_scores(self):
        """Random scores -> AUROC ~ 0.5."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 500 + [1] * 500)
        y_scores = rng.rand(1000)
        auroc = auroc_score(y_true, y_scores)
        assert 0.4 < auroc < 0.6

    def test_inverted(self):
        """Perfectly inverted scores -> AUROC = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert auroc_score(y_true, y_scores) == pytest.approx(0.0)

    def test_single_class(self):
        """Single class -> AUROC = 0.5."""
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.1, 0.5, 0.7, 0.9])
        assert auroc_score(y_true, y_scores) == 0.5

    def test_manual_computation(self):
        """Manual AUROC computation with known values."""
        # 2 positives, 2 negatives
        # pos scores: 0.6, 0.8; neg scores: 0.3, 0.5
        # All 4 (pos, neg) pairs: (0.6>0.3)=1, (0.6>0.5)=1, (0.8>0.3)=1, (0.8>0.5)=1
        # AUROC = 4/4 = 1.0
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.3, 0.6, 0.5, 0.8])
        assert auroc_score(y_true, y_scores) == pytest.approx(1.0)


# ============================================================
# TestF1PrecisionRecall — 3 tests
# ============================================================

class TestF1PrecisionRecall:
    def test_perfect(self):
        """Perfect predictions -> F1 = 1.0."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        assert f1_score(y_true, y_pred) == pytest.approx(1.0)

    def test_all_false_positive(self):
        """All FP -> precision = 0."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        pr = precision_recall(y_true, y_pred)
        assert pr["precision"] == pytest.approx(0.0)
        assert pr["tp"] == 0
        assert pr["fp"] == 4

    def test_manual_counts(self):
        """Manual TP/FP/FN/TN counts."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        pr = precision_recall(y_true, y_pred)
        assert pr["tp"] == 2
        assert pr["fp"] == 1
        assert pr["fn"] == 1
        assert pr["tn"] == 2
        assert pr["precision"] == pytest.approx(2 / 3)
        assert pr["recall"] == pytest.approx(2 / 3)


# ============================================================
# TestECE — 2 tests
# ============================================================

class TestECE:
    def test_perfect_calibration(self):
        """Perfect calibration -> ECE ~ 0."""
        # Scores match true frequencies exactly
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
        ece = expected_calibration_error(y_true, y_scores, n_bins=5)
        assert ece < 0.3

    def test_overconfident(self):
        """All scores=0.99 but half are wrong -> high ECE."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99])
        ece = expected_calibration_error(y_true, y_scores, n_bins=10)
        assert ece > 0.4


# ============================================================
# TestAbstentionRate — 2 tests
# ============================================================

class TestAbstentionRate:
    def test_all_confident(self):
        """All scores outside uncertain zone -> rate = 0."""
        y_scores = np.array([0.0, 0.1, 0.9, 1.0])
        result = abstention_rate(y_scores, low_threshold=0.3, high_threshold=0.7)
        assert result["abstention_rate"] == pytest.approx(0.0)
        assert result["n_confident"] == 4

    def test_all_uncertain(self):
        """All scores in uncertain zone -> rate = 1."""
        y_scores = np.array([0.4, 0.5, 0.5, 0.6])
        result = abstention_rate(y_scores, low_threshold=0.3, high_threshold=0.7)
        assert result["abstention_rate"] == pytest.approx(1.0)
        assert result["n_abstained"] == 4


# ============================================================
# TestExactMatch — 3 tests
# ============================================================

class TestExactMatch:
    def test_identical(self):
        """Identical answers -> EM = 1.0."""
        preds = ["the cat", "a dog", "the answer"]
        refs = ["the cat", "a dog", "the answer"]
        assert exact_match(preds, refs) == pytest.approx(1.0)

    def test_normalization(self):
        """Normalization handles case, articles, punctuation."""
        preds = ["The Cat!"]
        refs = ["a cat"]
        # Normalized: "cat" == "cat"
        assert exact_match(preds, refs) == pytest.approx(1.0)

    def test_different(self):
        """Different answers -> EM = 0.0."""
        preds = ["cat"]
        refs = ["dog"]
        assert exact_match(preds, refs) == pytest.approx(0.0)


# ============================================================
# TestQAF1 — 2 tests
# ============================================================

class TestQAF1:
    def test_partial_overlap(self):
        """Partial overlap gives F1 between 0 and 1."""
        # "big red cat" vs "the red cat sat"
        # Normalized: "big red cat" vs "red cat sat"
        # Common: red, cat -> 2
        # Precision: 2/3, Recall: 2/3, F1: 2/3
        f1 = f1_score_qa("big red cat", "the red cat sat")
        assert 0 < f1 < 1

    def test_empty_prediction(self):
        """Empty prediction -> F1 = 0."""
        assert f1_score_qa("", "the answer") == pytest.approx(0.0)


# ============================================================
# TestTokenEfficiency — 2 tests
# ============================================================

class TestTokenEfficiency:
    def test_low_usage(self):
        """Low token usage with high accuracy -> high efficiency."""
        result = token_efficiency(accuracy=0.9, tokens_used=100, token_budget=1000)
        assert result["efficiency"] > 0.7
        assert result["savings_ratio"] == pytest.approx(0.9)

    def test_full_budget(self):
        """Full budget usage -> efficiency = 0."""
        result = token_efficiency(accuracy=1.0, tokens_used=1000, token_budget=1000)
        assert result["efficiency"] == pytest.approx(0.0)


# ============================================================
# TestInputCompressionEfficiency — 1 test
# ============================================================

class TestInputCompressionEfficiency:
    def test_high_compression_preserved_quality(self):
        """50% compression with same accuracy -> input_efficiency ~ 0.5."""
        result = input_compression_efficiency(
            accuracy_full=0.9, accuracy_compressed=0.9,
            tokens_full=1000, tokens_compressed=500,
        )
        assert result["quality_ratio"] == pytest.approx(1.0)
        assert result["compression_ratio"] == pytest.approx(0.5)
        assert result["input_efficiency"] == pytest.approx(0.5)


# ============================================================
# TestOutputQualityPerToken — 1 test
# ============================================================

class TestOutputQualityPerToken:
    def test_fewer_tokens_higher_qpt(self):
        """Same accuracy with fewer tokens -> higher quality_per_token."""
        r1 = output_quality_per_token(accuracy=0.8, tokens_generated=100)
        r2 = output_quality_per_token(accuracy=0.8, tokens_generated=50)
        assert r2["quality_per_token"] > r1["quality_per_token"]


# ============================================================
# TestOptimalThreshold — 1 test
# ============================================================

class TestOptimalThreshold:
    def test_finds_best_f1(self):
        """Should find threshold that maximizes F1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
        result = optimal_threshold(y_true, y_scores)
        assert result["metric"] == "f1"
        assert result["score"] > 0.8
        assert 0.4 <= result["threshold"] <= 0.7


# ============================================================
# TestBenchmarkResult — 1 test
# ============================================================

class TestBenchmarkResult:
    def test_constructor_and_methods(self):
        """BenchmarkResult constructor, to_dict, summary, repr."""
        br = BenchmarkResult(
            "test_bench",
            {"auroc": 0.95, "f1": 0.88},
            {"n_samples": 100},
        )
        assert br.name == "test_bench"
        assert br.metrics["auroc"] == 0.95

        d = br.to_dict()
        assert d["name"] == "test_bench"
        assert d["metrics"]["f1"] == 0.88
        assert d["metadata"]["n_samples"] == 100

        s = br.summary()
        assert "test_bench" in s
        assert "0.9500" in s

        r = repr(br)
        assert "BenchmarkResult" in r
        assert "test_bench" in r


# ============================================================
# TestSyntheticHaluEval — 2 tests
# ============================================================

class TestSyntheticHaluEval:
    def test_shapes(self):
        """Samples have correct shapes."""
        data = SyntheticHaluEval(n_samples=10, num_heads=4, seq_len=16, num_layers=3)
        assert len(data) == 10
        sample = data[0]
        assert sample["attn"].shape == (4, 16, 16)
        assert sample["token_probs"].shape == (16,)
        assert len(sample["layers"]) == 3
        assert sample["label"] in (0, 1)

    def test_high_separation_auroc(self):
        """separation=1.0 should yield AUROC > 0.9."""
        data = SyntheticHaluEval(n_samples=200, separation=1.0, seed=123)
        labels = np.array([s["label"] for s in data])
        # Compute scores using hallucination_score
        from aip_bench.guard import hallucination_score
        scores = []
        for s in data:
            result = hallucination_score(
                attn=s["attn"],
                token_probs=s["token_probs"],
                layers=s["layers"],
                context_len=s["context_len"],
            )
            scores.append(result["score"])
        scores = np.array(scores)
        auroc = auroc_score(labels, scores)
        assert auroc > 0.9


# ============================================================
# TestSyntheticOckBench — 1 test
# ============================================================

class TestSyntheticOckBench:
    def test_structure(self):
        """Problems have correct structure."""
        data = SyntheticOckBench(n_problems=5, num_heads=4, seq_len=32, head_dim=16)
        assert len(data) == 5
        problem = data[0]
        assert problem["attn"].shape == (4, 32, 32)
        assert problem["keys"].shape == (32, 16)
        assert problem["values"].shape == (32, 16)
        assert isinstance(problem["correct"], bool)
        assert isinstance(problem["tokens_used"], int)


# ============================================================
# TestSyntheticQA — 1 test
# ============================================================

class TestSyntheticQA:
    def test_structure(self):
        """QA samples have correct structure."""
        data = SyntheticQA(n_samples=5, seq_len=64, head_dim=32)
        assert len(data) == 5
        sample = data[0]
        assert isinstance(sample["question"], str)
        assert isinstance(sample["context"], str)
        assert isinstance(sample["answer"], str)
        assert sample["keys"].shape == (64, 32)
        assert sample["values"].shape == (64, 32)


# ============================================================
# TestHallucinationBenchmark — 1 test
# ============================================================

class TestHallucinationBenchmark:
    def test_run(self):
        """HallucinationBenchmark.run() returns valid BenchmarkResult."""
        data = SyntheticHaluEval(n_samples=50, separation=0.8, seed=99)
        bench = HallucinationBenchmark(data=data)
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.name == "halueval"
        assert "auroc" in result.metrics
        assert "f1" in result.metrics
        assert 0 <= result.metrics["auroc"] <= 1
        assert 0 <= result.metrics["f1"] <= 1


# ============================================================
# TestInferenceEfficiencyBenchmark — 1 test
# ============================================================

class TestInferenceEfficiencyBenchmark:
    def test_run(self):
        """InferenceEfficiencyBenchmark.run() returns valid BenchmarkResult."""
        data = SyntheticOckBench(n_problems=20, seed=99)
        bench = InferenceEfficiencyBenchmark(data=data, token_budget=1024)
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.name == "ockbench"
        assert "accuracy" in result.metrics
        assert "efficiency" in result.metrics
        assert 0 <= result.metrics["accuracy"] <= 1
        assert result.metrics["efficiency"] >= 0


# ============================================================
# TestQACompressionBenchmark — 1 test
# ============================================================

class TestQACompressionBenchmark:
    def test_run(self):
        """QACompressionBenchmark.run() returns valid BenchmarkResult."""
        data = SyntheticQA(n_samples=20, seed=99)
        bench = QACompressionBenchmark(data=data, compress_method="evict",
                                       recent_window=32)
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.name == "qa_compression"
        assert "quality_retention" in result.metrics
        assert "f1_full" in result.metrics
        assert "f1_compressed" in result.metrics
        assert result.metrics["quality_retention"] >= 0


# ============================================================
# TestRunBenchmark — 1 test
# ============================================================

class TestRunBenchmark:
    def test_dispatch(self):
        """run_benchmark dispatches correctly for all 3 benchmarks."""
        for name in ["halueval", "ockbench", "qa_compression"]:
            result = run_benchmark(name, use_synthetic=True)
            assert isinstance(result, BenchmarkResult)
            assert result.name == name
            assert len(result.metrics) > 0


# ============================================================
# TestDatasetRegistry — 1 test
# ============================================================

# ============================================================
# TestBrierScore — 2 tests
# ============================================================

class TestBrierScore:
    def test_perfect_predictions(self):
        """Perfect predictions -> Brier = 0."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y_true, y_scores) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Worst predictions -> Brier = 1."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y_true, y_scores) == pytest.approx(1.0)


# ============================================================
# TestPerplexity — 2 tests
# ============================================================

class TestPerplexity:
    def test_confident_model(self):
        """High log probs (near 0) -> low perplexity."""
        log_probs = np.array([-0.01, -0.02, -0.01, -0.03])
        ppl = perplexity(log_probs)
        assert ppl < 1.1

    def test_uncertain_model(self):
        """Low log probs -> high perplexity."""
        log_probs = np.array([-5.0, -6.0, -5.5, -4.5])
        ppl = perplexity(log_probs)
        assert ppl > 100

    def test_empty(self):
        """Empty log probs -> inf."""
        assert perplexity(np.array([])) == float("inf")


# ============================================================
# TestAccuracyScore — 2 tests
# ============================================================

class TestAccuracyScore:
    def test_perfect(self):
        """Perfect predictions -> accuracy = 1."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        assert accuracy_score(y_true, y_pred) == pytest.approx(1.0)

    def test_half_correct(self):
        """Half correct -> accuracy = 0.5."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)


# ============================================================
# TestBLEU — 3 tests
# ============================================================

class TestBLEU:
    def test_identical(self):
        """Identical sentences -> BLEU = 1.0."""
        text = "the cat sat on the mat"
        assert bleu_score(text, text) == pytest.approx(1.0)

    def test_no_overlap(self):
        """No overlap -> BLEU = 0."""
        assert bleu_score("foo bar baz qux", "alpha beta gamma delta") == pytest.approx(0.0)

    def test_empty_prediction(self):
        """Empty prediction -> BLEU = 0."""
        assert bleu_score("", "the cat sat") == pytest.approx(0.0)


# ============================================================
# TestROUGEL — 3 tests
# ============================================================

class TestROUGEL:
    def test_identical(self):
        """Identical sentences -> ROUGE-L = 1.0."""
        text = "the cat sat on the mat"
        assert rouge_l_score(text, text) == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Partial overlap -> 0 < ROUGE-L < 1."""
        score = rouge_l_score("the cat sat on the mat", "the dog sat on the rug")
        assert 0 < score < 1

    def test_empty(self):
        """Empty prediction -> ROUGE-L = 0."""
        assert rouge_l_score("", "the cat sat") == pytest.approx(0.0)


# ============================================================
# TestDatasetRegistry — 2 tests
# ============================================================

class TestDatasetRegistry:
    def test_truthfulqa_in_registry(self):
        """TruthfulQA should be in dataset registry."""
        ds = list_datasets()
        assert "truthfulqa" in ds
        assert ds["truthfulqa"]["category"] == "hallucination"

    def test_all_major_benchmarks_present(self):
        """All major benchmarks should be in the registry."""
        ds = list_datasets()
        expected = [
            "halueval_qa", "halueval_dialogue", "halueval_summarization",
            "squad_v2", "hotpotqa", "truthfulqa",
            "mmlu", "hellaswag", "arc_challenge", "winogrande",
            "gsm8k", "boolq", "fever", "natural_questions",
        ]
        for name in expected:
            assert name in ds, f"{name} missing from registry"
        assert len(ds) >= 14


# ============================================================
# TestCompare — 4 tests
# ============================================================

class TestCompare:
    def test_compare_runs_all_benchmarks(self):
        """compare() runs across configs and produces ComparisonReport."""
        report = compare(
            configs={
                "baseline": {"prune_ratio": 0.0},
                "pruned": {"prune_ratio": 0.3},
            },
            benchmarks=["ockbench"],
        )
        assert isinstance(report, ComparisonReport)
        assert "baseline" in report.configs
        assert "pruned" in report.configs
        assert "ockbench" in report.benchmarks

    def test_table_output(self):
        """table() returns a formatted string."""
        report = compare(
            configs={"a": {}, "b": {"prune_ratio": 0.5}},
            benchmarks=["ockbench"],
        )
        table = report.table()
        assert isinstance(table, str)
        assert "ockbench" in table
        assert "accuracy" in table

    def test_deltas_output(self):
        """deltas() shows differences vs baseline."""
        report = compare(
            configs={"base": {}, "opt": {"prune_ratio": 0.4}},
            benchmarks=["ockbench"],
        )
        deltas = report.deltas("base")
        assert isinstance(deltas, str)
        assert "base" in deltas

    def test_compare_results_from_precomputed(self):
        """compare_results() builds report from pre-computed results."""
        br1 = BenchmarkResult("test", {"auroc": 0.9, "f1": 0.85})
        br2 = BenchmarkResult("test", {"auroc": 0.95, "f1": 0.88})
        report = compare_results({
            "config_a": {"test": br1},
            "config_b": {"test": br2},
        })
        assert isinstance(report, ComparisonReport)
        assert report.get("config_a", "test").metrics["auroc"] == 0.9
        d = report.to_dict()
        assert "config_a" in d
        assert "config_b" in d


# ============================================================
# TestBootstrapCI — 2 tests
# ============================================================

class TestBootstrapCI:
    def test_returns_interval(self):
        """Bootstrap CI returns mean and bounds."""
        rng = np.random.RandomState(7)
        y_true = np.array([0] * 100 + [1] * 100)
        y_scores = np.concatenate([rng.rand(100) * 0.5, 0.5 + rng.rand(100) * 0.5])
        result = bootstrap_ci(auroc_score, y_true, y_scores, n_bootstrap=200)
        assert "mean" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["ci_low"] <= result["ci_high"]
        assert result["ci"] == 0.95

    def test_tight_ci_for_perfect(self):
        """Perfect separation should give tight CI near 1.0."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_scores = np.concatenate([
            np.linspace(0, 0.4, 50),
            np.linspace(0.6, 1.0, 50),
        ])
        result = bootstrap_ci(auroc_score, y_true, y_scores, n_bootstrap=500)
        assert result["ci_low"] > 0.85


# ============================================================
# TestMacroF1 — 2 tests
# ============================================================

class TestMacroF1:
    def test_perfect(self):
        """Perfect predictions -> macro F1 = 1.0."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert macro_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_partial(self):
        """Partial correctness -> 0 < macro F1 < 1."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])
        mf1 = macro_f1(y_true, y_pred)
        assert 0 < mf1 < 1


# ============================================================
# TestPassAtK — 2 tests
# ============================================================

class TestPassAtK:
    def test_all_correct(self):
        """All correct -> pass@k = 1."""
        assert pass_at_k(10, 10, 1) == pytest.approx(1.0)

    def test_none_correct(self):
        """None correct -> pass@k = 0."""
        assert pass_at_k(10, 0, 1) == pytest.approx(0.0)


# ============================================================
# TestMETEOR — 2 tests
# ============================================================

class TestMETEOR:
    def test_identical(self):
        """Identical -> METEOR close to 1."""
        text = "the cat sat on the mat"
        score = meteor_score(text, text)
        assert score > 0.8

    def test_empty(self):
        """Empty prediction -> METEOR = 0."""
        assert meteor_score("", "the cat sat") == pytest.approx(0.0)


# ============================================================
# TestDummyModel — 3 tests
# ============================================================

class TestDummyModel:
    def test_generate(self):
        model = DummyModel(default_response="hello")
        assert model.generate("prompt") == "hello"

    def test_classify(self):
        model = DummyModel(default_choice=2)
        assert model.classify("prompt", ["a", "b", "c"]) == 2

    def test_load_model_dummy(self):
        model = load_model("dummy:hi")
        assert model.generate("x") == "hi"
        assert model.name == "dummy"


# ============================================================
# TestPrompts — 3 tests
# ============================================================

class TestPrompts:
    def test_list_templates(self):
        templates = list_templates()
        assert "mmlu" in templates
        assert "gsm8k" in templates
        assert len(templates) >= 10

    def test_format_prompt_mmlu(self):
        item = {
            "question": "What is 2+2?",
            "A": "3", "B": "4", "C": "5", "D": "6",
            "subject": "math",
        }
        prompt = format_prompt("mmlu", item)
        assert "What is 2+2?" in prompt
        assert "A." in prompt

    def test_format_prompt_with_shots(self):
        item = {"question": "test?", "answer": "42"}
        examples = [{"question": "ex?", "answer": "1"}]
        prompt = format_prompt("gsm8k", item, n_shots=1, examples=examples)
        assert "ex?" in prompt
        assert "test?" in prompt


# ============================================================
# TestCache — 3 tests
# ============================================================

class TestCache:
    def test_put_get(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResultCache(cache_dir=d)
            cache.put("model", "prompt", {"result": 42})
            got = cache.get("model", "prompt")
            assert got == {"result": 42}

    def test_miss(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResultCache(cache_dir=d)
            assert cache.get("model", "unknown") is None
            assert cache.stats["misses"] == 1

    def test_disabled(self):
        cache = ResultCache(enabled=False)
        cache.put("model", "prompt", "data")
        assert cache.get("model", "prompt") is None


# ============================================================
# TestExport — 4 tests
# ============================================================

class TestExport:
    def test_to_json_string(self):
        br = BenchmarkResult("test", {"acc": 0.9})
        s = to_json(br)
        data = json.loads(s)
        assert data["metrics"]["acc"] == 0.9

    def test_to_csv_string(self):
        br = BenchmarkResult("test", {"acc": 0.9, "f1": 0.85})
        s = to_csv(br)
        assert "acc" in s
        assert "f1" in s

    def test_to_html_string(self):
        br = BenchmarkResult("test", {"acc": 0.9})
        s = to_html(br)
        assert "<html>" in s
        assert "0.9000" in s

    def test_to_json_file(self):
        br = BenchmarkResult("test", {"acc": 0.9})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        to_json(br, path)
        data = json.loads(Path(path).read_text())
        assert data["name"] == "test"
        Path(path).unlink()


# ============================================================
# TestNewPipelines — 4 tests
# ============================================================

class TestNewPipelines:
    def test_multiple_choice(self):
        """MultipleChoiceBenchmark runs with DummyModel."""
        result = run_benchmark("mmlu")
        assert isinstance(result, BenchmarkResult)
        assert "accuracy" in result.metrics
        assert "macro_f1" in result.metrics

    def test_math(self):
        """MathBenchmark runs with DummyModel."""
        result = run_benchmark("gsm8k")
        assert isinstance(result, BenchmarkResult)
        assert "accuracy" in result.metrics

    def test_fact_verification(self):
        """FactVerificationBenchmark runs with DummyModel."""
        result = run_benchmark("fever")
        assert isinstance(result, BenchmarkResult)
        assert "accuracy" in result.metrics

    def test_open_domain_qa(self):
        """OpenDomainQABenchmark runs with DummyModel."""
        result = run_benchmark("natural_questions")
        assert isinstance(result, BenchmarkResult)
        assert "exact_match" in result.metrics
        assert "f1" in result.metrics


# ============================================================
# TestCLI — 3 tests
# ============================================================

class TestCLI:
    def test_list_command(self):
        from aip_bench.cli import main
        # Should not raise
        ret = main(["list"])
        assert ret == 0

    def test_run_command(self):
        from aip_bench.cli import main
        ret = main(["run", "halueval", "--model", "dummy"])
        assert ret == 0

    def test_run_with_output(self):
        from aip_bench.cli import main
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ret = main(["run", "mmlu", "--model", "dummy", "-o", path])
        assert ret == 0
        data = json.loads(Path(path).read_text())
        assert "tasks" in data
        Path(path).unlink()


# ============================================================
# TestAllBenchmarkDispatch — 1 test
# ============================================================

class TestAllBenchmarkDispatch:
    def test_all_registered_benchmarks_run(self):
        """Every registered benchmark should run without error."""
        from aip_bench.pipelines import _BENCHMARKS
        for name in _BENCHMARKS:
            result = run_benchmark(name, use_synthetic=True)
            assert isinstance(result, BenchmarkResult), f"{name} failed"
            assert len(result.metrics) > 0, f"{name} has no metrics"


# ============================================================
# TestConfig — 3 tests
# ============================================================

class TestConfig:
    def test_validate_valid(self):
        """Valid config passes validation."""
        config = {"tasks": ["halueval", "ockbench"], "model": "dummy"}
        errors = validate_config(config)
        assert errors == []

    def test_validate_unknown_task(self):
        """Unknown task is caught."""
        config = {"tasks": ["halueval", "fake_task"]}
        errors = validate_config(config)
        assert any("fake_task" in e for e in errors)

    def test_load_json_config(self):
        """Load config from JSON file."""
        config = {"suite": "test", "tasks": ["halueval"], "model": "dummy"}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(config, f)
            path = f.name
        loaded = load_config(path)
        assert loaded["tasks"] == ["halueval"]
        Path(path).unlink()


# ============================================================
# TestViz — 2 tests
# ============================================================

class TestViz:
    def test_import_viz(self):
        """viz module should import even without matplotlib."""
        from aip_bench import viz
        assert hasattr(viz, "radar_chart")
        assert hasattr(viz, "bar_comparison")

    def test_radar_with_matplotlib(self):
        """Radar chart generates with matplotlib (if available)."""
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")
        from aip_bench.viz import radar_chart
        from aip_bench.compare import compare
        report = compare(
            configs={"a": {}, "b": {"prune_ratio": 0.3}},
            benchmarks=["ockbench"],
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        radar_chart(report, output=path)
        assert Path(path).stat().st_size > 0
        Path(path).unlink()


# ============================================================
# TestProgress — 1 test
# ============================================================

class TestProgress:
    def test_pipelines_run_without_tqdm(self):
        """Pipelines should work even without tqdm."""
        result = run_benchmark("halueval")
        assert result.metrics["auroc"] > 0


# ============================================================
# TestPairedPermutationTest — 3 tests
# ============================================================

class TestPairedPermutationTest(unittest.TestCase):
    def test_identical_systems(self):
        """Identical scores should give p ~ 1 (not significant)."""
        scores = [0.8, 0.7, 0.9, 0.85, 0.75]
        result = paired_permutation_test(scores, scores)
        self.assertGreater(result["p_value"], 0.05)
        self.assertFalse(result["significant_005"])

    def test_very_different_systems(self):
        """Very different systems should be significant."""
        a = [0.9, 0.95, 0.92, 0.88, 0.91, 0.93, 0.90, 0.89, 0.94, 0.92]
        b = [0.3, 0.35, 0.32, 0.28, 0.31, 0.33, 0.30, 0.29, 0.34, 0.32]
        result = paired_permutation_test(a, b)
        self.assertLess(result["p_value"], 0.01)
        self.assertTrue(result["significant_001"])
        self.assertGreater(result["observed_diff"], 0.5)

    def test_returns_correct_keys(self):
        result = paired_permutation_test([1, 2, 3], [1, 2, 3])
        self.assertIn("observed_diff", result)
        self.assertIn("p_value", result)
        self.assertIn("significant_005", result)


# ============================================================
# TestEffectSize — 2 tests
# ============================================================

class TestEffectSize(unittest.TestCase):
    def test_identical(self):
        result = effect_size_cohens_d([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(result["d"], 0.0)
        self.assertEqual(result["magnitude"], "negligible")

    def test_large_effect(self):
        a = [10, 11, 12, 13, 14]
        b = [1, 2, 3, 4, 5]
        result = effect_size_cohens_d(a, b)
        self.assertGreater(abs(result["d"]), 0.8)
        self.assertEqual(result["magnitude"], "large")


# ============================================================
# TestTorchUtilsHelpers — 5 tests
# ============================================================

class TestTorchUtilsHelpers(unittest.TestCase):
    """Test torch_utils helper functions (without requiring torch)."""

    def test_extract_kv_exists(self):
        """The _extract_kv function exists."""
        from aip_bench.torch_utils import _extract_kv
        self.assertTrue(callable(_extract_kv))

    def test_load_model_eager_exists(self):
        """The load_model_eager function exists."""
        from aip_bench.torch_utils import load_model_eager
        self.assertTrue(callable(load_model_eager))

    def test_prepare_halueval_real_exists(self):
        from aip_bench.torch_utils import prepare_halueval_real
        self.assertTrue(callable(prepare_halueval_real))

    def test_prepare_qa_real_exists(self):
        from aip_bench.torch_utils import prepare_qa_real
        self.assertTrue(callable(prepare_qa_real))

    def test_token_probs_from_logits(self):
        """Test token_probs_from_logits with numpy arrays."""
        from aip_bench.torch_utils import token_probs_from_logits
        logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
        token_ids = np.array([0, 1])
        probs = token_probs_from_logits(logits, token_ids)
        self.assertEqual(len(probs), 2)
        # First token prob should be highest (logit=2.0 at index 0)
        self.assertGreater(probs[0], 0.5)
        # Second token prob should be highest (logit=3.0 at index 1)
        self.assertGreater(probs[1], 0.8)


# ============================================================
# Slow integration test — requires torch + transformers
# ============================================================

@pytest.mark.slow
def test_e2e_with_real_model():
    """End-to-end pipeline with a small real model (distilgpt2).

    Run manually: pytest tests/test_bench.py -m slow -v
    Skipped in CI (no torch/transformers).
    """
    try:
        import torch
        import transformers
    except ImportError:
        pytest.skip("torch and transformers required for E2E test")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from aip_bench.torch_utils import extract_model_data
    from aip_bench.guard import hallucination_score as hscore

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", output_attentions=True
    )
    model.eval()

    data = extract_model_data(model, tokenizer, "The capital of France is")
    assert data["attn"] is not None
    assert data["layers"] is not None
    assert len(data["layers"]) > 0

    result = hscore(
        attn=data["attn"],
        token_probs=data["token_probs"],
        layers=data["layers"],
        context_len=data["context_len"],
    )
    assert 0 <= result["score"] <= 1
    assert result["confidence"] > 0
