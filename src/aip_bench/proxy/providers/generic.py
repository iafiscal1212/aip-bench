"""Generic/fallback provider adapter."""

from .base import Provider


class GenericProvider(Provider):
    """Fallback provider — passthrough with optional message compression."""

    name = "generic"

    def build_url(self, target, path):
        if target:
            return target.rstrip("/") + path
        # Generic requires explicit target
        return None
