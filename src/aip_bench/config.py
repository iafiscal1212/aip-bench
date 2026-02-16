"""
Bench Config: YAML-based benchmark suite configuration.

Define benchmark suites in YAML files for reproducible evaluation.

Usage:
    from aip_bench.config import load_config, run_suite

    suite = load_config("bench_config.yaml")
    results = run_suite(suite)

Example bench_config.yaml:

    suite: "full_eval"
    model: "hf:distilgpt2"
    shots: 5
    tasks:
      - halueval
      - ockbench
      - mmlu
      - gsm8k
    configs:
      baseline:
        prune_ratio: 0.0
      optimized:
        prune_ratio: 0.25
    output:
      format: html
      path: results/report.html

Author: Carmen Esteban
"""

import json
from pathlib import Path

# YAML is optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(path):
    """Load a benchmark configuration from YAML or JSON.

    Parameters
    ----------
    path : str or Path
        Config file path (.yaml, .yml, or .json).

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path)
    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError(
                "PyYAML required for YAML configs. "
                "Install: pip install pyyaml"
            )
        return yaml.safe_load(text)
    elif path.suffix == ".json":
        return json.loads(text)
    else:
        # Try YAML first, then JSON
        if HAS_YAML:
            try:
                return yaml.safe_load(text)
            except Exception:
                pass
        return json.loads(text)


def run_suite(config):
    """Run a benchmark suite from a config dict.

    Parameters
    ----------
    config : dict
        Configuration with keys: suite, model, shots, tasks, configs, output.

    Returns
    -------
    dict
        Results: {task_name -> BenchmarkResult} or ComparisonReport.
    """
    from aip_bench.models import load_model
    from aip_bench.pipelines import run_benchmark
    from aip_bench.compare import compare
    from aip_bench.export import to_json, to_csv, to_html

    model_spec = config.get("model", "dummy")
    model = load_model(model_spec)
    shots = config.get("shots", 0)
    tasks = config.get("tasks", ["halueval", "ockbench", "qa_compression"])
    configs = config.get("configs")
    output_cfg = config.get("output", {})

    if configs and len(configs) > 1:
        # Multi-config comparison
        report = compare(configs=configs, benchmarks=tasks)
        result = report
    else:
        # Single run
        results = {}
        for task in tasks:
            kwargs = {}
            # Pass model/shots for new-style benchmarks
            if task not in ("halueval", "ockbench", "qa_compression"):
                kwargs["model"] = model
                kwargs["n_shots"] = shots
            results[task] = run_benchmark(task, **kwargs)
        result = results

    # Export if configured
    out_path = output_cfg.get("path")
    out_fmt = output_cfg.get("format", "json")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_fmt == "json":
            to_json(result if hasattr(result, "to_dict") else
                    _results_to_exportable(result), out_path)
        elif out_fmt == "csv":
            to_csv(result if hasattr(result, "results") else
                   next(iter(result.values())), out_path)
        elif out_fmt == "html":
            to_html(result if hasattr(result, "results") else
                    _results_to_exportable(result), out_path)

    return result


def _results_to_exportable(results):
    """Convert dict of BenchmarkResults to something exportable."""
    from aip_bench.compare import compare_results
    if isinstance(results, dict):
        # Wrap in a single "run" config
        return compare_results({"run": results})
    return results


def validate_config(config):
    """Validate a benchmark configuration.

    Parameters
    ----------
    config : dict
        Configuration to validate.

    Returns
    -------
    list of str
        Validation errors (empty if valid).
    """
    from aip_bench.pipelines import _BENCHMARKS

    errors = []

    if not isinstance(config, dict):
        return ["Config must be a dictionary"]

    tasks = config.get("tasks", [])
    if not tasks:
        errors.append("No tasks specified")
    for task in tasks:
        if task not in _BENCHMARKS:
            errors.append(f"Unknown task: {task!r}")

    model = config.get("model", "dummy")
    if ":" not in model and model != "dummy":
        errors.append(f"Invalid model spec: {model!r} (use backend:name)")

    output = config.get("output", {})
    if output.get("format") and output["format"] not in ("json", "csv", "html"):
        errors.append(f"Unknown output format: {output['format']!r}")

    return errors
