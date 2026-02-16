"""
CLI entry point for aip-proxy.

Usage:
    aip-proxy --port 8080 --profile balanced
    aip-proxy --port 8080 --profile aggressive --verbose
    aip-proxy --port 8080 --target http://localhost:11434
"""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="aip-proxy",
        description="Accordion-powered LLM proxy. Compresses chat messages in real-time.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Compression profile (default: balanced)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Override target API URL (e.g., http://localhost:11434)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args(argv)

    try:
        from .server import ProxyServer
    except ImportError:
        print(
            "Error: aiohttp is required for aip-proxy.\n"
            "Install with: pip install aip-bench[proxy]",
            file=sys.stderr,
        )
        sys.exit(1)

    server = ProxyServer(
        port=args.port,
        profile=args.profile,
        target=args.target,
        verbose=args.verbose,
    )
    server.run()
