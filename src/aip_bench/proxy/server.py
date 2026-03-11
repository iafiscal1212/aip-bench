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

    def __init__(self, port=8080, profile="balanced", target=None, verbose=True):
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
        # Special status page
        if request.path == "/_stats":
            return self._render_stats()

        # Read raw body
        try:
            body = await request.json()
        except json.JSONDecodeError:
            # Non-JSON request — passthrough raw
            raw = await request.read()
            return await self._forward_raw(request, raw)

        # Detect provider
        provider = detect_provider(request.path, dict(request.headers))
        logger.debug("[%s] %s → provider=%s", request.method, request.path, provider.name)

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
            return await self._forward(target_url, body, headers, provider.name)

    def _render_stats(self):
        """Render a simple HTML dashboard with compression stats."""
        s = self.stats.summary()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIP Proxy Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #f4f7f6; color: #333; line-height: 1.6; margin: 0; padding: 20px; }}
                .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 20px; max-width: 600px; margin: 20px auto; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 0; }}
                .stat-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }}
                .stat-item {{ background: #f9f9f9; padding: 15px; border-radius: 6px; border-left: 4px solid #3498db; }}
                .stat-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; font-weight: bold; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .savings {{ color: #27ae60; }}
                .footer {{ text-align: center; font-size: 12px; color: #95a5a6; margin-top: 20px; }}
            </style>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <div class="card">
                <h1>AIP Bench Proxy</h1>
                <p>Profile: <strong>{self.accordion.profile_name.upper()}</strong> | Uptime: {s['elapsed_seconds']}s</p>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Total Requests</div>
                        <div class="stat-value">{s['requests']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Compressions</div>
                        <div class="stat-value">{s['compressions']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Tokens Saved</div>
                        <div class="stat-value savings">{s['tokens_saved']:,}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Savings %</div>
                        <div class="stat-value savings">{s['savings_pct']:.1f}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Input Tokens</div>
                        <div class="stat-value">{s['tokens_before_total']:,}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Output Tokens</div>
                        <div class="stat-value">{s['tokens_after_total']:,}</div>
                    </div>
                </div>
            </div>
            <div class="footer">AIP-Bench • Efficiency Through Context Compression</div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    async def _forward(self, url, body, headers, provider_name="unknown"):
        """Forward request and return JSON response."""
        session = await self._get_session()
        async with session.post(url, json=body, headers=headers) as upstream:
            data = await upstream.read()
            if upstream.status >= 400:
                try:
                    err_text = data.decode("utf-8", errors="replace")
                except Exception:
                    err_text = "<unreadable>"
                logger.debug(
                    "[%s] Upstream error %d: %s",
                    provider_name, upstream.status, err_text[:1000],
                )
            else:
                logger.debug("[%s] Upstream %d OK", provider_name, upstream.status)
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
            if upstream.status >= 400:
                try:
                    err_text = data.decode("utf-8", errors="replace")
                except Exception:
                    err_text = "<unreadable>"
                logger.debug("Upstream error %d (raw): %s", upstream.status, err_text[:1000])
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
            logger.debug("Stream opened → upstream %d", upstream.status)
            if upstream.status >= 400:
                data = await upstream.read()
                try:
                    err_text = data.decode("utf-8", errors="replace")
                except Exception:
                    err_text = "<unreadable>"
                logger.debug("Upstream stream error %d: %s", upstream.status, err_text[:1000])
                await response.write(data)
            else:
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
            "[%s] Compressed: %d → %d tokens (%.1f%% saved)",
            provider_name, before, stats.get("tokens_after", 0), pct,
        )
        if self.verbose:
            logger.info(
                "  Messages: %d → %d",
                stats.get("messages_before"), stats.get("messages_after"),
            )

    async def _on_shutdown(self, _app):
        """Cleanup on server shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
        summary = self.stats.summary()
        logger.info("Session summary: %s", summary)

    def run(self):
        """Start the proxy server."""
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.handle_request)
        app.on_shutdown.append(self._on_shutdown)

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            force=True,
        )
        logger.info(
            "aip-proxy starting on port %d (profile=%s, target=%s)",
            self.port, self.accordion.profile_name, self.target or "auto",
        )
        web.run_app(app, port=self.port, print=None)
