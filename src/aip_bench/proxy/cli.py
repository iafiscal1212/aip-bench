"""
CLI entry point for aip-proxy.

Usage:
    aip-proxy --port 8080 --profile balanced
    aip-proxy --port 8080 --profile aggressive
    aip-proxy --port 8080 --target http://localhost:11434
    aip-proxy --port 8080 --no-verbose
"""

import argparse

from .server import ProxyServer


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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose/debug logging (default: on, use --no-verbose to disable)",
    )
    args = parser.parse_args(argv)

    server = ProxyServer(
        port=args.port,
        profile=args.profile,
        target=args.target,
        verbose=args.verbose,
    )
    server.run()
