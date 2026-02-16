"""
HTTP proxy server using aiohttp.

Sits between any LLM client and any API provider, compressing messages
in real-time using the Accordion engine. Supports streaming SSE passthrough.
"""

import json
import logging

try:
    from aiohttp import web, ClientSession
except ImportError:
    web = None
    ClientSession = None

from .accordion import MessageAccordion
from .providers import detect_provider
from .stats import CompressionStats

logger = logging.getLogger("aip-proxy")


class ProxyServer:
    """Accordion-powered LLM proxy server.

    Args:
        port: Port to listen on.
        profile: Compression profile name.
        target: Optional override target API URL.
        verbose: Enable verbose logging.
    """

    def __init__(self, port=8080, profile="balanced", target=None, verbose=False):
        if web is None:
            raise ImportError(
                "aiohttp is required for the proxy server. "
                "Install with: pip install aip-bench[proxy]"
            )

        self.port = port
        self.target = target
        self.verbose = verbose
        self.accordion = MessageAccordion(profile=profile)
        self.stats = CompressionStats()
        self.accordion.stats = self.stats
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = ClientSession()
        return self._session

    async def handle_request(self, request):
        """Main request handler: detect, compress, forward."""
        # Read raw body
        try:
            body = await request.json()
        except json.JSONDecodeError:
            # Non-JSON request — passthrough raw
            raw = await request.read()
            return await self._forward_raw(request, raw)

        # Detect provider
        provider = detect_provider(request.path, dict(request.headers))

        # Extract and compress messages
        messages = provider.extract_messages(body)
        if messages:
            compressed, comp_stats = self.accordion.compress(
                messages, model=body.get("model")
            )
            body = provider.replace_messages(body, compressed)

            if comp_stats.get("compressed"):
                self._log_savings(comp_stats, provider.name)
            else:
                self.stats.record_passthrough()
        else:
            self.stats.record_passthrough()

        # Build upstream URL
        target_url = provider.build_url(self.target, request.path)
        if not target_url:
            return web.json_response(
                {"error": "No target URL. Use --target or a recognized API path."},
                status=502,
            )

        # Forward headers
        headers = provider.forward_headers(request.headers)

        # Stream or regular forward
        if body.get("stream"):
            return await self._stream(target_url, body, headers, request)
        else:
            return await self._forward(target_url, body, headers)

    async def _forward(self, url, body, headers):
        """Forward request and return JSON response."""
        session = await self._get_session()
        async with session.post(url, json=body, headers=headers) as upstream:
            data = await upstream.read()
            return web.Response(
                body=data,
                status=upstream.status,
                content_type=upstream.content_type,
            )

    async def _forward_raw(self, request, raw_body):
        """Forward a non-JSON request as-is."""
        if not self.target:
            return web.json_response(
                {"error": "No target URL configured for raw passthrough."},
                status=502,
            )
        session = await self._get_session()
        url = self.target.rstrip("/") + request.path
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        async with session.request(
            request.method, url, data=raw_body, headers=headers
        ) as upstream:
            data = await upstream.read()
            return web.Response(
                body=data,
                status=upstream.status,
                content_type=upstream.content_type,
            )

    async def _stream(self, url, body, headers, request):
        """Stream SSE response transparently."""
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)

        session = await self._get_session()
        async with session.post(url, json=body, headers=headers) as upstream:
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)

        await response.write_eof()
        return response

    def _log_savings(self, stats, provider_name):
        """Log compression savings."""
        saved = stats.get("tokens_saved", 0)
        before = stats.get("tokens_before", 0)
        pct = (saved / before * 100) if before > 0 else 0
        logger.info(
            f"[{provider_name}] Compressed: {before} -> {stats.get('tokens_after', 0)} "
            f"tokens ({pct:.1f}% saved)"
        )
        if self.verbose:
            logger.info(f"  Messages: {stats.get('messages_before')} -> {stats.get('messages_after')}")

    async def _on_shutdown(self, app):
        """Cleanup on server shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
        summary = self.stats.summary()
        logger.info(f"Session summary: {summary}")

    def run(self):
        """Start the proxy server."""
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.handle_request)
        app.on_shutdown.append(self._on_shutdown)

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
        )
        logger.info(
            f"aip-proxy starting on port {self.port} "
            f"(profile={self.accordion.profile_name}, target={self.target or 'auto'})"
        )
        web.run_app(app, port=self.port, print=None)
