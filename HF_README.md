---
license: mit
language:
  - en
tags:
  - benchmarking
  - hallucination-detection
  - kv-cache-compression
  - llm-evaluation
  - attention-analysis
library_name: aip-bench
pipeline_tag: text-classification
---

# AIP Bench

Benchmarking suite for LLM hallucination detection, inference efficiency, and QA compression evaluation.

## Key Results (distilgpt2)

### Hallucination Detection
| Data Source | AUROC | F1 |
|-------------|-------|-----|
| Synthetic | 1.000 | 1.000 |
| HaluEval QA (real) | 0.788 | 0.767 |

### KV Cache Compression
| Dataset | Quality Retention | Token Savings |
|---------|-------------------|---------------|
| SQuAD v2 (real) | 91.7% | ~78% |
| HotPotQA (real) | 86.0% | ~78% |
| Synthetic | 88.2% | ~78% |

### Compression Profiles
| Profile | Token Savings | Quality Retention |
|---------|---------------|-------------------|
| Conservative | ~74% | ~100% |
| Balanced | ~78% | ~88% |
| Aggressive | ~85% | ~77% |

## Install

```bash
pip install -e .                 # Core (numpy only)
pip install -e ".[bench-full]"   # + torch + transformers + datasets
```

## Quick Start

```python
from aip.bench import run_benchmark

result = run_benchmark('halueval')
print(result.summary())
```

## Features

- 20+ metrics (AUROC, F1, ECE, Brier, BLEU, ROUGE-L, METEOR, Bootstrap CI, pass@k)
- 14 benchmark datasets (HaluEval, MMLU, GSM8K, SQuAD, HotPotQA, FEVER, etc.)
- 4 model backends (HuggingFace, OpenAI, Anthropic, Dummy)
- CLI: `aip-bench run`, `compare`, `list`, `export`
- Synthetic + real data evaluation
- All metrics are numpy-pure (no sklearn)

## Author

Carmen Esteban — [GitHub](https://github.com/iafiscal1212/experimentos)
