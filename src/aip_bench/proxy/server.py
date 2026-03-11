"""
HTTP proxy server using aiohttp.

Sits between any LLM client and any API provider, compressing messages
in real-time using the Accordion engine. Supports streaming SSE passthrough.
"""

import json
import logging

from aiohttp import web, ClientSession

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
                messages, provider=provider, model=body.get("model")
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
        pct = s['savings_pct']
        # Color based on savings
        color = "#27ae60" if pct > 50 else "#f39c12" if pct > 20 else "#3498db"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIP Proxy Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; line-height: 1.6; margin: 0; padding: 20px; }}
                .container {{ max-width: 800px; margin: 40px auto; }}
                .card {{ background: #1e293b; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); padding: 30px; border: 1px solid #334155; }}
                h1 {{ color: #38bdf8; margin-top: 0; font-weight: 300; letter-spacing: -1px; display: flex; justify-content: space-between; align-items: center; }}
                .badge {{ background: #334155; color: #94a3b8; font-size: 12px; padding: 4px 12px; border-radius: 20px; font-weight: bold; text-transform: uppercase; }}
                .profile-tag {{ color: #38bdf8; font-weight: bold; }}
                
                .progress-container {{ background: #334155; border-radius: 10px; height: 20px; margin: 25px 0; overflow: hidden; position: relative; }}
                .progress-bar {{ background: {color}; height: 100%; width: {pct}%; transition: width 1s ease-in-out; border-radius: 10px; }}
                .progress-text {{ position: absolute; width: 100%; text-align: center; top: 0; line-height: 20px; font-size: 11px; font-weight: bold; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }}
                
                .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 30px; }}
                .stat-item {{ background: #0f172a; padding: 20px; border-radius: 10px; border: 1px solid #334155; transition: transform 0.2s; }}
                .stat-item:hover {{ transform: translateY(-5px); border-color: #38bdf8; }}
                .stat-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: bold; letter-spacing: 1px; }}
                .stat-value {{ font-size: 28px; font-weight: bold; color: #f1f5f9; margin-top: 5px; }}
                .stat-value.savings {{ color: {color}; }}
                
                .footer {{ text-align: center; font-size: 12px; color: #64748b; margin-top: 40px; border-top: 1px solid #334155; padding-top: 20px; }}
                .uptime {{ font-style: italic; }}
            </style>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <div class="container">
                <div class="card">
                    <h1>
                        AIP Proxy Dashboard
                        <span class="badge">Live Stats</span>
                    </h1>
                    <p>Optimization Profile: <span class="profile-tag">{self.accordion.profile_name.upper()}</span> | <span class="uptime">Uptime: {s['elapsed_seconds']}s</span></p>
                    
                    <div class="progress-container">
                        <div class="progress-bar"></div>
                        <div class="progress-text">TOTAL TOKEN SAVINGS: {pct:.1f}%</div>
                    </div>

                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-label">Requests</div>
                            <div class="stat-value">{s['requests']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Tokens Saved</div>
                            <div class="stat-value savings">{s['tokens_saved']:,}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Ratio</div>
                            <div class="stat-value">{s['avg_compression_ratio']:.2f}x</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">In (Context)</div>
                            <div class="stat-value">{s['tokens_before_total']:,}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Out (Network)</div>
                            <div class="stat-value">{s['tokens_after_total']:,}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Compressions</div>
                            <div class="stat-value">{s['compressions']}</div>
                        </div>
                    </div>
                </div>
                <div class="footer">
                    AIP-Bench Engine • Accurate Tiktoken Counting • {self.accordion.profile_name.capitalize()} Mode
                </div>
            </div>
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
