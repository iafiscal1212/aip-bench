"""
AIP Bench: Benchmarking for hallucination detection and inference optimization.

Evaluate AIP's guard and inference modules against standard benchmarks:
- HaluEval/HalluLens: Hallucination detection (AUROC, F1)
- OckBench: Reasoning efficiency (token efficiency)
- SQuAD/HotPotQA: QA quality under KV compression (EM, F1 retention)
- MMLU/HellaSwag/ARC/WinoGrande: Multiple-choice knowledge & reasoning
- GSM8K: Math reasoning
- FEVER/TruthfulQA: Fact verification
- NQ: Open-domain QA

Quick start:
    from aip_bench import run_benchmark

    # Run with synthetic data (no downloads needed)
    result = run_benchmark('halueval')
    print(result.summary())

    # Run all benchmarks
    for name in ['halueval', 'ockbench', 'mmlu', 'gsm8k']:
        result = run_benchmark(name)
        print(result)

CLI:
    aip-bench run halueval mmlu gsm8k --model dummy
    aip-bench compare --configs base:prune_ratio=0 opt:prune_ratio=0.5
    aip-bench list

Author: Carmen Esteban
"""

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
    qa_metrics,
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
from aip_bench.datasets import (
    list_datasets,
    load_dataset,
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
from aip_bench.models import (
    BaseModel,
    DummyModel,
    load_model,
)
from aip_bench.prompts import (
    format_prompt,
    list_templates,
    get_template,
)
from aip_bench.cache import ResultCache
from aip_bench.export import to_json, to_csv, to_html
from aip_bench.config import load_config, run_suite, validate_config
from aip_bench.logging_utils import get_logger, setup_file_logging, quiet, verbose
from aip_bench.proxy import MessageAccordion, CompressionStats, detect_provider

__all__ = [
    # Evaluator — metrics
    "auroc_score",
    "f1_score",
    "precision_recall",
    "expected_calibration_error",
    "abstention_rate",
    "brier_score",
    "perplexity",
    "accuracy_score",
    "bleu_score",
    "rouge_l_score",
    "exact_match",
    "f1_score_qa",
    "qa_metrics",
    "token_efficiency",
    "input_compression_efficiency",
    "output_quality_per_token",
    "optimal_threshold",
    "bootstrap_ci",
    "macro_f1",
    "pass_at_k",
    "meteor_score",
    "paired_permutation_test",
    "effect_size_cohens_d",
    "BenchmarkResult",
    # Datasets
    "list_datasets",
    "load_dataset",
    "SyntheticHaluEval",
    "SyntheticOckBench",
    "SyntheticQA",
    # Pipelines
    "run_benchmark",
    "HallucinationBenchmark",
    "InferenceEfficiencyBenchmark",
    "QACompressionBenchmark",
    "MultipleChoiceBenchmark",
    "MathBenchmark",
    "FactVerificationBenchmark",
    "OpenDomainQABenchmark",
    # Compare
    "compare",
    "compare_results",
    "ComparisonReport",
    # Models
    "BaseModel",
    "DummyModel",
    "load_model",
    # Prompts
    "format_prompt",
    "list_templates",
    "get_template",
    # Cache
    "ResultCache",
    # Export
    "to_json",
    "to_csv",
    "to_html",
    # Config
    "load_config",
    "run_suite",
    "validate_config",
    # Logging
    "get_logger",
    "setup_file_logging",
    "quiet",
    "verbose",
    # Proxy
    "MessageAccordion",
    "CompressionStats",
    "detect_provider",
]
