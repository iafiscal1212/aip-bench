"""
Fallback HTTP proxy server using only stdlib (http.server + urllib).

Used when aiohttp is not available (e.g., Colab, minimal environments).
Does not support streaming SSE — buffers the full response before returning.
"""

import json
import logging
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

from .accordion import MessageAccordion
from .providers import detect_provider
from .stats import CompressionStats

logger = logging.getLogger("aip-proxy")


class _ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that compresses and forwards LLM requests."""

    def do_POST(self):
        self._handle()

    def do_PUT(self):
        self._handle()

    def do_GET(self):
        self._handle()

    def _handle(self):
        accordion = self.server.accordion
        stats = self.server.proxy_stats
        target = self.server.target

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        # Try JSON parse
        body = None
        try:
            if raw_body:
                body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        provider = detect_provider(self.path, dict(self.headers))
        logger.debug("[%s] %s → provider=%s", self.command, self.path, provider.name)

        if body is not None:
            messages = provider.extract_messages(body)
            if messages:
                compressed, comp_stats = accordion.compress(
                    messages, model=body.get("model")
                )
                body = provider.replace_messages(body, compressed)
                if comp_stats.get("compressed"):
                    saved = comp_stats.get("tokens_saved", 0)
                    before = comp_stats.get("tokens_before", 0)
                    pct = (saved / before * 100) if before > 0 else 0
                    logger.info(
                        "[%s] Compressed: %d → %d tokens (%.1f%% saved)",
                        provider.name, before, comp_stats.get("tokens_after", 0), pct,
                    )
                else:
                    stats.record_passthrough()
            else:
                stats.record_passthrough()
            forward_body = json.dumps(body).encode()
        else:
            stats.record_passthrough()
            forward_body = raw_body

        # Build upstream URL
        target_url = provider.build_url(target, self.path)
        if not target_url:
            self._send_json(502, {"error": "No target URL. Use --target or a recognized API path."})
            return

        # Build headers
        fwd_headers = provider.forward_headers(dict(self.headers))
        fwd_headers["Content-Length"] = str(len(forward_body))

        # Forward
        req = urllib.request.Request(
            target_url,
            data=forward_body,
            headers=fwd_headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                resp_body = resp.read()
                logger.debug("[%s] Upstream %d OK", provider.name, resp.status)
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            try:
                err_text = resp_body.decode("utf-8", errors="replace")
            except Exception:
                err_text = "<unreadable>"
            logger.error("[%s] Upstream error %d: %s", provider.name, e.code, err_text[:1000])
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.URLError as e:
            logger.error("Upstream connection failed: %s", e.reason)
            self._send_json(502, {"error": f"Upstream connection failed: {e.reason}"})

    def _send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.debug("%s - " + format, self.address_string(), *args)


class StdlibProxyServer:
    """Proxy server using only stdlib. Fallback when aiohttp is unavailable.

    Note: does not support SSE streaming (buffers full response).
    """

    def __init__(self, port=8080, profile="balanced", target=None, verbose=True):
        self.port = port
        self.target = target
        self.verbose = verbose
        self.accordion = MessageAccordion(profile=profile)
        self.stats = CompressionStats()
        self.accordion.stats = self.stats

    def run(self):
        server = HTTPServer(("0.0.0.0", self.port), _ProxyHandler)
        server.accordion = self.accordion
        server.proxy_stats = self.stats
        server.target = self.target
        server.verbose = self.verbose

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        )
        logger.info(
            "aip-proxy (stdlib) starting on port %d (profile=%s, target=%s)",
            self.port, self.accordion.profile_name, self.target or "auto",
        )
        logger.info("Note: stdlib server does not support SSE streaming")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            summary = self.stats.summary()
            logger.info("Session summary: %s", summary)
