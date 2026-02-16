"""
Bench CLI: Command-line interface for aip-bench.

Usage:
    aip-bench run halueval ockbench mmlu --model dummy --shots 0
    aip-bench run gsm8k --model hf:distilgpt2 --output results.json
    aip-bench compare --configs base:prune_ratio=0 opt:prune_ratio=0.5 --tasks ockbench
    aip-bench list
    aip-bench export results.json --format html --output report.html

Author: Carmen Esteban
"""

import argparse
import json
import sys
import time


def main(argv=None):
    """Entry point for aip-bench CLI."""
    parser = argparse.ArgumentParser(
        prog="aip-bench",
        description="AIP Bench — Benchmarking for LLM evaluation",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── run ──────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run benchmarks")
    p_run.add_argument(
        "tasks", nargs="+",
        help="Benchmark names (e.g. halueval mmlu gsm8k)",
    )
    p_run.add_argument(
        "--model", "-m", default="dummy",
        help="Model spec (dummy, hf:distilgpt2, openai:gpt-4o, "
             "anthropic:claude-sonnet-4-5-20250929)",
    )
    p_run.add_argument("--shots", "-s", type=int, default=0, help="Few-shot count")
    p_run.add_argument("--output", "-o", help="Output file (json/csv/html)")
    p_run.add_argument("--format", "-f", default="json",
                       choices=["json", "csv", "html"],
                       help="Output format")
    p_run.add_argument("--prune-ratio", type=float, default=0.25,
                       help="Prune ratio for inference benchmarks")
    p_run.add_argument("--token-budget", type=int, default=1024,
                       help="Token budget for efficiency benchmarks")

    # ── compare ─────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare configurations")
    p_cmp.add_argument(
        "--configs", nargs="+", required=True,
        help="Configs as name:key=val (e.g. base:prune_ratio=0 opt:prune_ratio=0.5)",
    )
    p_cmp.add_argument(
        "--tasks", nargs="+", default=["halueval", "ockbench", "qa_compression"],
        help="Benchmarks to compare across",
    )
    p_cmp.add_argument("--baseline", default=None,
                       help="Baseline config name for deltas")
    p_cmp.add_argument("--output", "-o", help="Output file")
    p_cmp.add_argument("--format", "-f", default="json",
                       choices=["json", "csv", "html"])

    # ── list ────────────────────────────────────────────
    sub.add_parser("list", help="List available benchmarks and datasets")

    # ── export ──────────────────────────────────────────
    p_exp = sub.add_parser("export", help="Convert results between formats")
    p_exp.add_argument("input", help="Input JSON file")
    p_exp.add_argument("--format", "-f", default="html",
                       choices=["json", "csv", "html"])
    p_exp.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "compare":
        return _cmd_compare(args)
    elif args.command == "list":
        return _cmd_list(args)
    elif args.command == "export":
        return _cmd_export(args)
    return 0


def _cmd_run(args):
    """Run benchmarks."""
    from aip_bench.models import load_model
    from aip_bench.pipelines import run_benchmark, _BENCHMARKS
    from aip_bench.export import to_json, to_csv, to_html

    model = load_model(args.model)
    results = {}

    for task in args.tasks:
        if task not in _BENCHMARKS:
            print(f"Unknown benchmark: {task}", file=sys.stderr)
            continue

        t0 = time.time()
        print(f"Running {task}...", end=" ", flush=True)

        kwargs = {}
        # Pass model and shots for new-style benchmarks
        if task not in ("halueval", "ockbench", "qa_compression"):
            kwargs["model"] = model
            kwargs["n_shots"] = args.shots
        else:
            if task == "ockbench":
                kwargs["prune_ratio"] = args.prune_ratio
                kwargs["token_budget"] = args.token_budget

        result = run_benchmark(task, **kwargs)
        elapsed = time.time() - t0
        results[task] = result

        # Print summary
        headline = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(result.metrics.items())[:4]
        )
        print(f"done ({elapsed:.1f}s) — {headline}")

    # Export
    if args.output:
        # Wrap in a dict for multi-task export
        from aip_bench.evaluator import BenchmarkResult
        export_data = {
            "tasks": {
                name: br.to_dict() for name, br in results.items()
            },
            "model": args.model,
            "shots": args.shots,
        }
        if args.format == "json":
            with open(args.output, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif args.format == "csv":
            # Export first result for CSV
            if results:
                first = next(iter(results.values()))
                to_csv(first, args.output)
        elif args.format == "html":
            if len(results) == 1:
                first = next(iter(results.values()))
                to_html(first, args.output)
            else:
                # Build a simple comparison report
                from aip_bench.compare import compare_results
                report = compare_results({"run": results})
                to_html(report, args.output)
        print(f"Results saved to {args.output}")

    return 0


def _cmd_compare(args):
    """Compare configurations."""
    from aip_bench.compare import compare
    from aip_bench.export import to_json, to_csv, to_html

    configs = {}
    for spec in args.configs:
        parts = spec.split(":", maxsplit=1)
        name = parts[0]
        kwargs = {}
        if len(parts) > 1:
            for pair in parts[1].split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    # Try numeric conversion
                    try:
                        v = float(v)
                        if v == int(v):
                            v = int(v)
                    except ValueError:
                        pass
                    kwargs[k] = v
        configs[name] = kwargs

    print(f"Comparing {list(configs.keys())} across {args.tasks}...")
    t0 = time.time()
    report = compare(configs, benchmarks=args.tasks)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s)\n")

    # Print table
    print(report.table())

    # Print deltas if baseline specified
    baseline = args.baseline or list(configs.keys())[0]
    if len(configs) > 1:
        print(report.deltas(baseline))

    # Export
    if args.output:
        if args.format == "json":
            to_json(report, args.output)
        elif args.format == "csv":
            to_csv(report, args.output)
        elif args.format == "html":
            to_html(report, args.output)
        print(f"Results saved to {args.output}")

    return 0


def _cmd_list(args):
    """List available benchmarks and datasets."""
    from aip_bench.pipelines import _BENCHMARKS
    from aip_bench.datasets import list_datasets

    print("Available benchmarks:")
    print("=" * 50)
    for name in sorted(_BENCHMARKS.keys()):
        cls = _BENCHMARKS[name]
        print(f"  {name:<25s}  {cls.__name__}")

    print()
    print("Available datasets (require 'datasets' library):")
    print("=" * 50)
    for name, info in sorted(list_datasets().items()):
        print(f"  {name:<25s}  [{info['category']}] {info['description']}")

    return 0


def _cmd_export(args):
    """Convert results between formats."""
    from aip_bench.evaluator import BenchmarkResult
    from aip_bench.export import to_json, to_csv, to_html

    with open(args.input) as f:
        data = json.load(f)

    # Reconstruct BenchmarkResult if possible
    if "tasks" in data:
        results = {}
        for task_name, task_data in data["tasks"].items():
            results[task_name] = BenchmarkResult(
                task_data.get("name", task_name),
                task_data.get("metrics", {}),
                task_data.get("metadata", {}),
            )
        from aip_bench.compare import compare_results
        report = compare_results({"run": results})
        obj = report
    elif "metrics" in data:
        obj = BenchmarkResult(
            data.get("name", ""),
            data.get("metrics", {}),
            data.get("metadata", {}),
        )
    else:
        print("Unrecognized input format", file=sys.stderr)
        return 1

    output = args.output or f"output.{args.format}"
    if args.format == "json":
        to_json(obj, output)
    elif args.format == "csv":
        to_csv(obj, output)
    elif args.format == "html":
        to_html(obj, output)

    print(f"Exported to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
