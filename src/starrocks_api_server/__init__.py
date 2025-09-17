"""StarRocks HTTP API server package."""

from .app import create_app, app

__all__ = ["create_app", "app"]
