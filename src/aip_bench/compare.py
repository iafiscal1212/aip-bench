"""
Bench Compare: Run multiple configurations and produce delta tables.

Compares baseline vs optimized (or any A vs B) across all benchmarks,
showing absolute values and deltas for every metric.

Usage:
    from aip_bench.compare import compare, ComparisonReport

    report = compare(
        configs={
            "baseline": {"prune_ratio": 0.0},
            "pruned_25": {"prune_ratio": 0.25},
            "pruned_50": {"prune_ratio": 0.50},
        },
        benchmarks=["halueval", "ockbench", "qa_compression"],
    )
    print(report.table())
    print(report.deltas("baseline"))

Author: Carmen Esteban
"""

import numpy as np

from aip_bench.evaluator import BenchmarkResult
from aip_bench.pipelines import run_benchmark


class ComparisonReport:
    """Container for multi-configuration benchmark comparison.

    Parameters
    ----------
    results : dict
        Mapping of config_name -> {benchmark_name -> BenchmarkResult}.
    """

    def __init__(self, results):
        self.results = results

    @property
    def configs(self):
        return list(self.results.keys())

    @property
    def benchmarks(self):
        if not self.results:
            return []
        first = next(iter(self.results.values()))
        return list(first.keys())

    def get(self, config, benchmark):
        """Get a single BenchmarkResult."""
        return self.results[config][benchmark]

    def table(self):
        """Generate a full comparison table as a string.

        Returns
        -------
        str
            Formatted table with all configs x benchmarks x metrics.
        """
        lines = []
        for bench_name in self.benchmarks:
            lines.append(f"=== {bench_name} ===")
            # Collect all metric names
            all_metrics = set()
            for cfg in self.configs:
                br = self.results[cfg].get(bench_name)
                if br:
                    all_metrics.update(br.metrics.keys())
            all_metrics = sorted(all_metrics)

            # Header
            cfg_width = max(len(c) for c in self.configs)
            header = f"  {'metric':<30s}"
            for cfg in self.configs:
                header += f"  {cfg:>{cfg_width}s}"
            lines.append(header)
            lines.append("  " + "-" * (30 + (cfg_width + 2) * len(self.configs)))

            # Rows
            for metric in all_metrics:
                row = f"  {metric:<30s}"
                for cfg in self.configs:
                    br = self.results[cfg].get(bench_name)
                    val = br.metrics.get(metric, None) if br else None
                    if isinstance(val, float):
                        row += f"  {val:>{cfg_width}.4f}"
                    elif val is not None:
                        row += f"  {str(val):>{cfg_width}s}"
                    else:
                        row += f"  {'—':>{cfg_width}s}"
                lines.append(row)
            lines.append("")

        return "\n".join(lines)

    def deltas(self, baseline):
        """Generate delta table relative to a baseline config.

        Parameters
        ----------
        baseline : str
            Name of the baseline configuration.

        Returns
        -------
        str
            Formatted table showing (value, delta, %change) vs baseline.
        """
        if baseline not in self.results:
            raise ValueError(f"Baseline {baseline!r} not in configs: {self.configs}")

        others = [c for c in self.configs if c != baseline]
        lines = []

        for bench_name in self.benchmarks:
            lines.append(f"=== {bench_name} (vs {baseline}) ===")
            base_br = self.results[baseline].get(bench_name)
            if not base_br:
                lines.append("  (no baseline data)")
                continue

            all_metrics = sorted(base_br.metrics.keys())
            # Header
            header = f"  {'metric':<25s}  {baseline:>10s}"
            for cfg in others:
                header += f"  {cfg:>10s}  {'delta':>8s}  {'%':>7s}"
            lines.append(header)
            lines.append("  " + "-" * (25 + 12 + len(others) * 30))

            for metric in all_metrics:
                base_val = base_br.metrics.get(metric)
                if not isinstance(base_val, float):
                    continue
                row = f"  {metric:<25s}  {base_val:>10.4f}"
                for cfg in others:
                    other_br = self.results[cfg].get(bench_name)
                    other_val = other_br.metrics.get(metric) if other_br else None
                    if isinstance(other_val, float):
                        delta = other_val - base_val
                        pct = (delta / abs(base_val) * 100) if abs(base_val) > 1e-12 else 0.0
                        sign = "+" if delta >= 0 else ""
                        row += f"  {other_val:>10.4f}  {sign}{delta:>7.4f}  {sign}{pct:>6.1f}%"
                    else:
                        row += f"  {'—':>10s}  {'—':>8s}  {'—':>7s}"
                lines.append(row)
            lines.append("")

        return "\n".join(lines)

    def to_dict(self):
        """Convert entire report to nested dict."""
        out = {}
        for cfg, benchmarks in self.results.items():
            out[cfg] = {}
            for bench_name, br in benchmarks.items():
                out[cfg][bench_name] = br.to_dict()
        return out

    def metric_summary(self):
        """One-line-per-config summary across all benchmarks.

        Returns
        -------
        str
            Compact summary.
        """
        lines = []
        for cfg in self.configs:
            parts = [f"{cfg}:"]
            for bench_name in self.benchmarks:
                br = self.results[cfg].get(bench_name)
                if br:
                    # Pick the "headline" metric per benchmark type
                    headline = _headline_metric(br)
                    if headline:
                        parts.append(f"{bench_name}={headline}")
            lines.append("  ".join(parts))
        return "\n".join(lines)

    def __repr__(self):
        return (f"ComparisonReport(configs={self.configs}, "
                f"benchmarks={self.benchmarks})")


def _headline_metric(br):
    """Pick the most important metric from a BenchmarkResult."""
    m = br.metrics
    for key in ["auroc", "efficiency", "quality_retention", "accuracy", "f1"]:
        if key in m and isinstance(m[key], float):
            return f"{m[key]:.4f}"
    return None


def compare(configs, benchmarks=None, **shared_kwargs):
    """Run benchmarks across multiple configurations and compare.

    Parameters
    ----------
    configs : dict
        Mapping of config_name -> dict of kwargs for run_benchmark.
        Example: {"baseline": {"prune_ratio": 0.0}, "pruned": {"prune_ratio": 0.5}}
    benchmarks : list of str, optional
        Which benchmarks to run. Default: all three.
    **shared_kwargs
        Additional kwargs applied to all configs.

    Returns
    -------
    ComparisonReport
        Full comparison with table() and deltas() methods.

    Example
    -------
    >>> report = compare(
    ...     configs={"baseline": {}, "aggressive": {"prune_ratio": 0.5}},
    ...     benchmarks=["ockbench"],
    ... )
    >>> print(report.deltas("baseline"))
    """
    if benchmarks is None:
        benchmarks = ["halueval", "ockbench", "qa_compression"]

    results = {}
    for cfg_name, cfg_kwargs in configs.items():
        results[cfg_name] = {}
        merged = {**shared_kwargs, **cfg_kwargs}
        for bench_name in benchmarks:
            # Filter kwargs to avoid passing irrelevant params to benchmarks
            try:
                br = run_benchmark(bench_name, **merged)
            except TypeError:
                # Fallback: run without config kwargs if they don't apply
                br = run_benchmark(bench_name, **shared_kwargs)
            results[cfg_name][bench_name] = br

    return ComparisonReport(results)


def compare_results(results_dict):
    """Build a ComparisonReport from pre-computed BenchmarkResults.

    Parameters
    ----------
    results_dict : dict
        Mapping of config_name -> {benchmark_name -> BenchmarkResult}.

    Returns
    -------
    ComparisonReport
    """
    return ComparisonReport(results_dict)
