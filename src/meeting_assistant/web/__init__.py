"""Browser UI for the pipeline: ``meeting-assistant serve``.

``jobs`` runs submissions in a background worker and persists everything to
disk; ``app`` exposes a single page and a small JSON API over it.
"""

from .app import create_app

__all__ = ["create_app"]
