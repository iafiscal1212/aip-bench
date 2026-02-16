"""
Bench Viz: Visualization for benchmark results.

Generates radar charts, bar comparisons, and exports to PNG/SVG.
Requires matplotlib (optional dependency).

Usage:
    from aip_bench.viz import radar_chart, bar_comparison

    radar_chart(report, metrics=["auroc", "f1", "accuracy"], output="radar.png")
    bar_comparison(report, metric="efficiency", output="bars.png")

Author: Carmen Esteban
"""

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _require_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib required for visualization. "
            "Install: pip install matplotlib"
        )


def radar_chart(report, metrics=None, output=None, title="Benchmark Comparison",
                figsize=(8, 8)):
    """Radar chart comparing configs across metrics.

    Parameters
    ----------
    report : ComparisonReport
        Comparison report with multiple configs.
    metrics : list of str, optional
        Metrics to plot. If None, auto-selects from first benchmark.
    output : str, optional
        Output file path (png/svg/pdf). If None, returns figure.
    title : str
        Chart title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure if output is None.
    """
    _require_matplotlib()

    # Collect metrics from first benchmark
    bench_name = report.benchmarks[0] if report.benchmarks else None
    if bench_name is None:
        raise ValueError("Report has no benchmarks")

    if metrics is None:
        # Auto-select float metrics
        first_br = report.results[report.configs[0]].get(bench_name)
        if first_br:
            metrics = [k for k, v in first_br.metrics.items()
                       if isinstance(v, float)][:8]
        else:
            metrics = []

    if not metrics:
        raise ValueError("No metrics to plot")

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(report.configs)))

    for i, cfg in enumerate(report.configs):
        br = report.results[cfg].get(bench_name)
        if not br:
            continue
        values = [br.metrics.get(m, 0.0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=cfg,
                color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def bar_comparison(report, metric="accuracy", output=None,
                   title=None, figsize=(10, 6)):
    """Bar chart comparing a single metric across configs and benchmarks.

    Parameters
    ----------
    report : ComparisonReport
        Comparison report.
    metric : str
        Metric to compare.
    output : str, optional
        Output file path. If None, returns figure.
    title : str, optional
        Chart title. Auto-generated if None.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    _require_matplotlib()

    configs = report.configs
    benchmarks = report.benchmarks
    n_configs = len(configs)
    n_benchmarks = len(benchmarks)

    if title is None:
        title = f"{metric} across configurations"

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_benchmarks)
    width = 0.8 / max(n_configs, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for i, cfg in enumerate(configs):
        values = []
        for bench in benchmarks:
            br = report.results[cfg].get(bench)
            val = br.metrics.get(metric, 0.0) if br else 0.0
            values.append(val if isinstance(val, (int, float)) else 0.0)
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=cfg, color=colors[i])
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Benchmark")
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def metric_heatmap(report, output=None, title="Metric Heatmap",
                   figsize=(12, 8)):
    """Heatmap of all metrics across configs for first benchmark.

    Parameters
    ----------
    report : ComparisonReport
        Comparison report.
    output : str, optional
        Output file path.
    title : str
        Chart title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    _require_matplotlib()

    bench_name = report.benchmarks[0] if report.benchmarks else None
    if bench_name is None:
        raise ValueError("Report has no benchmarks")

    configs = report.configs
    all_metrics = set()
    for cfg in configs:
        br = report.results[cfg].get(bench_name)
        if br:
            all_metrics.update(
                k for k, v in br.metrics.items() if isinstance(v, float)
            )
    metrics = sorted(all_metrics)

    data = np.zeros((len(configs), len(metrics)))
    for i, cfg in enumerate(configs):
        br = report.results[cfg].get(bench_name)
        if br:
            for j, m in enumerate(metrics):
                data[i, j] = br.metrics.get(m, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)

    # Annotate cells
    for i in range(len(configs)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if data[i, j] > 0.6 else "black")

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
